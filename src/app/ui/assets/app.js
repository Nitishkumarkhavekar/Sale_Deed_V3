/* ==========================================================================
   Sale Deed AI - webview client.

   Talks to Python over QWebChannel. Every call is asynchronous: the UI thread
   must never wait on OCR, inference or a database query, because blocking it
   freezes the window - which is precisely what the specification forbids.

   Live values (AI status, GPU, queue, progress) are patched into the existing
   DOM rather than re-rendering the page. Re-rendering during a running batch
   would flash the screen every few seconds and throw away scroll position.
   ========================================================================== */

(function () {
  "use strict";

  let api = null;                 // Python bridge object
  let currentPage = "dashboard";
  let pollTimer = null;
  //: Last failed-OCR count seen by the poll. `null` means "not yet known", which
  //: is distinct from 0 - without that distinction the first poll after load
  //: looks like a change and refreshes the page out from under the operator.
  let lastFailedOcr = null;
  const POLL_MS = 2500;

  // In-flight calls, keyed by request id. Python replies on the `completed`
  // signal carrying that id, which is how an asynchronous result finds the
  // promise that is waiting for it.
  const pending = {};
  let requestSeq = 0;
  //: Generous: an export of a large batch or a watermark scan legitimately
  //: takes a while. This exists to stop a lost reply leaking a promise, not to
  //: bound normal work.
  const CALL_TIMEOUT_MS = 120000;

  // ---------------------------------------------------------------- bridge

  function connect() {
    return new Promise(function (resolve) {
      if (typeof QWebChannel === "undefined" || !window.qt || !qt.webChannelTransport) {
        // Rendered outside the shell (preview in a browser). Everything still
        // displays; only actions are inert.
        console.warn("QWebChannel unavailable - preview mode, actions disabled");
        resolve(null);
        return;
      }
      new QWebChannel(qt.webChannelTransport, function (channel) {
        api = channel.objects.bridge;
        // Results arrive on a signal keyed by request id, not through a
        // callback argument. QWebChannel strips a trailing function and uses it
        // as the reply handler, so a slot that declares one is never matched -
        // Qt logs `No candidates found for "render" with 1 arguments` and the
        // call is silently dropped.
        api.completed.connect(function (id, raw) {
          const waiter = pending[id];
          if (!waiter) return;          // superseded, or the page moved on
          delete pending[id];
          clearTimeout(waiter.timer);
          let parsed;
          try { parsed = JSON.parse(raw); }
          catch (e) { parsed = { ok: false, error: "malformed reply: " + String(raw).slice(0, 200) }; }
          if (parsed && parsed.ok === false) waiter.reject(new Error(parsed.error || "failed"));
          else waiter.resolve(parsed);
        });
        resolve(api);
      });
    });
  }

  function call(method, payload) {
    return new Promise(function (resolve, reject) {
      if (!api || typeof api[method] !== "function") {
        reject(new Error("bridge method unavailable: " + method));
        return;
      }
      const id = String(++requestSeq);
      // A reply that never arrives would leave the promise pending forever and
      // the entry in `pending` leaked. Time it out and say so.
      const timer = setTimeout(function () {
        if (pending[id]) {
          delete pending[id];
          reject(new Error(method + " timed out after " + (CALL_TIMEOUT_MS / 1000) + "s"));
        }
      }, CALL_TIMEOUT_MS);
      pending[id] = { resolve: resolve, reject: reject, timer: timer };
      try {
        api[method](id, JSON.stringify(payload || {}));
      } catch (err) {
        delete pending[id];
        clearTimeout(timer);
        reject(err);
      }
    });
  }

  // ------------------------------------------------------------- utilities

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  function toast(message, level) {
    const host = $("#content");
    if (!host) return;
    const el = document.createElement("div");
    el.className = "notice " + (level || "info");
    el.innerHTML = message;
    host.insertBefore(el, host.firstChild);
    if (level !== "danger") setTimeout(function () { el.remove(); }, 6000);
  }

  function busy(button, on) {
    if (!button) return;
    button.disabled = !!on;
    if (on) {
      button.dataset.label = button.innerHTML;
      button.innerHTML = '<span class="spinner"></span>';
    } else if (button.dataset.label) {
      button.innerHTML = button.dataset.label;
      delete button.dataset.label;
    }
  }

  // ------------------------------------------------------------ navigation

  function pageFromHash() {
    const h = (location.hash || "#dashboard").replace(/^#/, "");
    return h.split("?")[0] || "dashboard";
  }

  function navigate(page, params) {
    currentPage = page;
    call("render", { page: page, params: params || {} })
      .then(function (res) {
        if (!res || !res.html) return;
        // Replace the content area, never the document.
        //
        // `document.write` tears down this JavaScript context and re-runs this
        // file, which constructs a *second* QWebChannel over the same
        // transport. The new channel takes over `onmessage` with an empty
        // `execCallbacks`, so every reply still in flight for the old one -
        // most often the 2.5 s status poll - arrives with an id it has never
        // seen and JavaScript throws
        // `channel.execCallbacks[message.id] is not a function`.
        //
        // One channel per session is the only arrangement in which an
        // asynchronous reply can be relied on to arrive.
        const host = $("#content");
        if (!host) return;
        host.innerHTML = res.html;
        setActiveNav(page);
        // Delegated handlers survive this, but anything bound to a specific
        // element does not.
        bindDropzone();
        // The Download card is rebuilt by every render, so its enabled state
        // has to be re-derived rather than assumed.
        syncDownloadState();
        window.scrollTo(0, 0);
      })
      .catch(function (err) { toast("Could not open " + page + ": " + err.message, "danger"); });
  }

  function setActiveNav(page) {
    $$(".sidenav a, nav a").forEach(function (link) {
      const href = link.getAttribute("href") || "";
      link.classList.toggle("active", href.slice(1) === page);
    });
    syncTabs(page);
  }

  // The two top tabs are coarser than the left navigation: "Data View" is one
  // page, "Control Panel" is everything else. Highlighting neither while the
  // operator is on Settings would read as a broken control, so Control Panel
  // stays lit for any page that is not the Data View.
  function syncTabs(page) {
    $$(".tabs a").forEach(function (link) {
      const target = (link.getAttribute("href") || "").slice(1);
      const on = target === "data" ? page === "data" : page !== "data";
      link.classList.toggle("active", on);
    });
  }

  // ------------------------------------------------------- live status poll

  function applyStatus(s) {
    if (!s) return;

    const dot = $("#ai-status .dot");
    if (dot) dot.className = "dot " + (s.ai_ready ? "on" : "off");
    const txt = $("#ai-status-text");
    if (txt) txt.textContent = "AI Server " + (s.ai_ready ? "Active" : "Offline");

    if ($("#gpu-util")) $("#gpu-util").textContent = s.gpu_util || "-";
    if ($("#vram-free")) $("#vram-free").textContent = s.vram_free || "-";
    if ($("#queue-text")) $("#queue-text").textContent = s.runner_state || "idle";
    const qdot = $("#queue-status .dot");
    if (qdot) qdot.className = "dot " + (s.running ? "busy" : "off");

    const start = $("#btn-start"), pause = $("#btn-pause"), stop = $("#btn-stop");
    if (start) start.disabled = !!s.running;
    if (pause) pause.disabled = !s.running;
    if (stop) stop.disabled = !s.running;

    // Patch progress bars in place so the page never flashes mid-batch.
    if (s.stages) {
      Object.keys(s.stages).forEach(function (key) {
        const st = s.stages[key];
        const row = $('[data-stage="' + key + '"]');
        if (!row) return;
        const fill = $(".bar > span", row);
        if (fill) fill.style.width = st.percent + "%";
        const count = $(".count", row);
        if (count) count.textContent = st.done + " / " + st.total + " · " + st.percent + "%";
      });
    }
    if (s.counts) {
      Object.keys(s.counts).forEach(function (key) {
        const el = $('[data-count="' + key + '"]');
        if (el) el.textContent = s.counts[key];
      });
    }
    // Failed-OCR badge in the nav. Hidden at zero rather than showing "0",
    // which reads as a category that exists and is empty; the point of the
    // badge is to be noticed only when there is something to notice.
    const badge = $("#nav-failed-count");
    if (badge && typeof s.failed_ocr === "number") {
      badge.textContent = s.failed_ocr;
      badge.style.display = s.failed_ocr > 0 ? "" : "none";
    }
    // While the Failed OCR page is open, a retry that succeeds removes a row.
    // Refresh on any change to the count so the list cannot show a file that
    // has already passed.
    if (currentPage === "failed_ocr" && typeof s.failed_ocr === "number" &&
        s.failed_ocr !== lastFailedOcr && lastFailedOcr !== null) {
      lastFailedOcr = s.failed_ocr;
      navigate("failed_ocr");
      return;
    }
    if (typeof s.failed_ocr === "number") lastFailedOcr = s.failed_ocr;

    // A finished batch changes more than numbers; a full refresh is warranted.
    if (s.batch_completed) navigate(currentPage);
  }

  function poll() {
    if (!api) return;
    call("status", {})
      .then(applyStatus)
      .catch(function () { /* transient; the next tick retries */ });
  }

  function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, POLL_MS);
    poll();
  }

  // ----------------------------------------------------------- interactions

  function bindGlobal() {
    document.addEventListener("click", function (ev) {
      const t = ev.target.closest("button, a, .dropzone, [data-collapsible] > .head");
      if (!t) return;

      // Collapsible sections (prompt, FAQs)
      if (t.matches("[data-collapsible] > .head")) {
        t.parentElement.classList.toggle("open");
        return;
      }

      // Nav
      if (t.tagName === "A" && t.getAttribute("href") &&
          t.getAttribute("href").charAt(0) === "#") {
        ev.preventDefault();
        location.hash = t.getAttribute("href");
        navigate(t.getAttribute("href").slice(1));
        return;
      }

      const d = t.dataset || {};

      if (t.id === "btn-start" || t.id === "btn-pause" || t.id === "btn-stop") {
        const action = t.id.replace("btn-", "");
        busy(t, true);
        call("control", { action: action })
          .then(function (r) { busy(t, false); applyStatus(r.status); })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-theme") { toggleTheme(); return; }

      if (t.id === "btn-refresh" || t.id === "btn-apply") { navigate(currentPage, collectFilters()); return; }

      // -- failed OCR ------------------------------------------------------
      //
      // One handler for one file, the selection, and all of them: the only
      // difference is which payload goes to `rerun_ocr`, so branching here
      // rather than in three near-identical blocks keeps the reporting and the
      // error handling identical for all three.
      if (t.id === "btn-refresh-failed") { navigate("failed_ocr"); return; }

      // Revalidate: ask the backend what the file is *now*. This is how a
      // repaired document stops being marked non-retryable - nothing is
      // queued, because reprocessing a file the operator just swapped in
      // should be their decision, not a side effect.
      if (t.id === "btn-revalidate-all" || t.classList.contains("btn-revalidate")) {
        const payload = t.id === "btn-revalidate-all"
          ? (t.dataset.batchId && t.dataset.batchId !== "0"
              ? { batch_id: t.dataset.batchId } : {})
          : { document_pk: +t.dataset.documentPk };
        busy(t, true);
        call("revalidate", payload)
          .then(function (r) {
            busy(t, false);
            toast(r.detail || "Re-checked.", r.repaired ? "ok" : "info");
            navigate("failed_ocr");
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.classList.contains("btn-open-file")) {
        call("open_path", { path: t.dataset.path })
          .catch(function (e) { toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-export-corrupted") {
        busy(t, true);
        call("export_corrupted", t.dataset.batchId && t.dataset.batchId !== "0"
             ? { batch_id: t.dataset.batchId } : {})
          .then(function (r) {
            busy(t, false);
            toast(r.rows
              ? "Wrote " + r.rows + " corrupted file(s) to <code>" + r.path + "</code>"
              : "No corrupted PDFs to report.", r.rows ? "ok" : "info");
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-rerun-all" || t.id === "btn-rerun-selected" ||
          t.classList.contains("btn-rerun-one")) {
        let payload;
        if (t.id === "btn-rerun-all") {
          payload = { all: true };
          // Scope it to the batch the page is showing, when it is showing one.
          // "0" is the model's "no batch"; sending it would be a real id.
          if (t.dataset.batchId && t.dataset.batchId !== "0") {
            payload.batch_id = t.dataset.batchId;
          }
        } else if (t.id === "btn-rerun-selected") {
          payload = { document_pks: selectedFailed() };
          if (!payload.document_pks.length) { toast("Select at least one file.", "warn"); return; }
        } else {
          payload = { document_pk: +t.dataset.documentPk };
        }
        busy(t, true);
        call("rerun_ocr", payload)
          .then(function (r) {
            busy(t, false);
            // count 0 is not an error - the list was stale, or someone else
            // requeued them first. The service says which, and saying nothing
            // would look like the button is broken.
            toast(r.detail || (r.count + " document(s) queued."),
                  r.count ? "ok" : "warn");
            navigate("failed_ocr");
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }
      if (d.goto) { location.hash = "#" + d.goto; navigate(d.goto); return; }
      if (d.page) { navigate(currentPage, Object.assign(collectFilters(), { page: +d.page })); return; }

      if (d.viewBatch) { openModal("batch_detail", { batch_id: +d.viewBatch }); return; }
      if (d.viewDocument) { openModal("document_detail", { document_pk: +d.viewDocument }); return; }
      if (t.matches("[data-modal-close]")) { closeModal(); return; }

      if (d.exportBatch || d.exportFailed) {
        busy(t, true);
        call("export", { batch_id: +(d.exportBatch || d.exportFailed),
                         failed_only: !!d.exportFailed })
          .then(function (r) { busy(t, false); toast("Exported " + r.rows + " rows to <code>" + r.path + "</code>", "ok"); })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (d.reprocessBatch) {
        if (!confirm("Requeue every failed document in this batch?")) return;
        busy(t, true);
        call("reprocess", { batch_id: +d.reprocessBatch })
          .then(function (r) { busy(t, false); toast("Requeued " + r.count + " document(s).", "ok"); navigate(currentPage); })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (d.deleteBatch) {
        if (!confirm("Delete this batch and all of its extracted data? This cannot be undone.")) return;
        call("delete_batch", { batch_id: +d.deleteBatch })
          .then(function () { toast("Batch deleted.", "ok"); navigate(currentPage); })
          .catch(function (e) { toast(e.message, "danger"); });
        return;
      }

      // The zone says "click to select", so it has to select. Each dropzone
      // opens the dialog its own page owns: the watermark page keeps a separate
      // selection, and routing its zone at the upload picker would stage files
      // for *processing* instead of cleaning.
      if (t.classList && t.classList.contains("dropzone")) {
        if (t.id === "wm-dropzone") {
          call("watermark", { action: "browse" })
            .then(function () { navigate(currentPage); })
            .catch(function (e) { toast(e.message, "danger"); });
        } else {
          call("pick_files", {})
            .then(function (r) { if (r.count) navigate(currentPage); })
            .catch(function (e) { toast(e.message, "danger"); });
        }
        return;
      }

      if (t.id === "btn-browse") {
        call("pick_files", {})
          .then(function (r) { if (r.count) navigate(currentPage); })
          .catch(function (e) { toast(e.message, "danger"); });
        return;
      }

      // The watermark page keeps its own selection. Routing its buttons at the
      // upload selection would queue the files for *processing* instead, which
      // is a different and destructive-feeling surprise.
      if (t.id === "btn-wm-browse" || t.id === "btn-wm-scan" ||
          t.id === "btn-wm-remove" || t.id === "btn-wm-open" ||
          t.id === "btn-wm-clear") {
        const action = t.id.slice("btn-wm-".length);
        busy(t, true);
        call("watermark", { action: action })
          .then(function (r) {
            busy(t, false);
            if (action === "scan" && r.scanned === 0) {
              toast("Nothing to scan - choose some PDFs first.", "warn");
            } else if (action === "remove") {
              toast(r.removed
                ? "Cleaned " + r.removed + " file(s) into " + r.output_dir
                : "No removable watermarks were found.",
                r.removed ? "ok" : "warn");
            } else if (action === "open" && r.path) {
              call("open_path", { path: r.path })
                .catch(function () { toast("Cleaned copies are in " + r.path, "info"); });
            }
            if (action !== "open") navigate(currentPage);
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-add-batch") { submitBatch(t); return; }
      if (t.id === "btn-clear") {
        call("clear_selection", {}).then(function () { navigate(currentPage); });
        return;
      }
      if (d.viewDocument) {
        // The button has existed since the Data View was written and never had
        // a handler - it uses a data- attribute, which the dead-control audit
        // only checked for id= attributes.
        openModal("pdf_viewer", { document_pk: +d.viewDocument });
        return;
      }

      if (t.id === "btn-copy-page-text") {
        const pk = +t.dataset.document;
        busy(t, true);
        call("document_text", { document_pk: pk })
          .then(function (r) {
            busy(t, false);
            if (!r.text) { toast(r.detail, "warn"); return; }
            // The clipboard API needs a secure origin; app:// is registered as
            // a secure scheme, so this works. The textarea fallback covers a
            // preview opened outside the shell.
            const done = function () { toast("Copied " + r.detail + ".", "ok"); };
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(r.text).then(done).catch(function () {
                legacyCopy(r.text); done();
              });
            } else { legacyCopy(r.text); done(); }
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-save-settings") { saveSettings(t); return; }
      if (t.id === "btn-save-rules") { saveRules(t); return; }

      // Download: pick a batch, optionally pick a location, then download.
      //
      // The location dialog is a separate step because it must run on the GUI
      // thread, while the export itself runs on the bridge pool - a thousand
      // document batch is written without the window freezing.
      if (t.id === "btn-choose-location") {
        const select = $("#download-batch");
        const label = select && select.value
          ? select.options[select.selectedIndex].dataset.name || "export"
          : "export";
        const scope = ($("#download-scope") || {}).value === "failed";
        const suggested = label.replace(/[^A-Za-z0-9._-]+/g, "_") +
          (scope ? "_failed" : "") + ".csv";
        call("pick_save_path", { suggested: suggested })
          .then(function (chosen) {
            if (!chosen || chosen.cancelled) return;
            downloadLocation = chosen.path;
            const shown = $("#download-location");
            if (shown) shown.textContent = chosen.path;
            syncDownloadState();
          })
          .catch(function (e) { toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-download-batch") {
        const select = $("#download-batch");
        const batchId = select && select.value;
        if (!batchId) { downloadStatus("Select a batch first.", "warn"); return; }
        const name = select.options[select.selectedIndex].dataset.name || "batch";
        const scope = ($("#download-scope") || {}).value === "failed";

        busy(t, true);
        downloadStatus("Downloading " + name +
                       " - this can take a moment for a large batch.");
        call("export_view", {
          batch_id: batchId,
          failed_only: scope,
          destination: downloadLocation || "",
        })
          .then(function (r) {
            busy(t, false);
            downloadStatus("Download complete: " + (r.rows || 0) + " row(s) from " +
                           (r.batch_name || name) + " saved to " + r.path, "ok");
            toast("Downloaded " + (r.rows || 0) + " row(s) from " +
                  (r.batch_name || name) + " to <code>" + r.path + "</code>", "ok");
          })
          .catch(function (e) {
            busy(t, false);
            downloadStatus("Download failed: " + e.message, "danger");
            toast(e.message, "danger");
          });
        return;
      }

      if (t.id === "btn-export-view") {
        busy(t, true);
        call("export_view", collectFilters())
          .then(function (r) {
            busy(t, false);
            toast("Exported " + (r.rows || 0) + " row(s) to " + r.path, "ok");
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-reset-prompt") {
        // The prompt is what the model was finetuned against, so replacing it
        // is not a cosmetic change - confirm before overwriting.
        if (!window.confirm("Restore the extraction prompt to its shipped " +
                            "default? Your edits will be lost.")) return;
        busy(t, true);
        call("reset_prompt", {})
          .then(function () {
            busy(t, false);
            toast("Prompt restored to the shipped default.", "ok");
            navigate(currentPage);
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-check-updates") {
        busy(t, true);
        call("check_updates", {})
          .then(function (r) {
            busy(t, false);
            toast(r.detail, r.url ? "info" : "warn");
            if (r.url) call("open_path", { path: r.url }).catch(function () {});
          })
          .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        return;
      }

      if (t.id === "btn-edit-prompt") {
        const ta = $("#prompt");
        if (!ta) return;
        if (ta.readOnly) {
          if (!confirm("The model was finetuned on this exact prompt. Editing it can " +
                       "reduce accuracy or break the output schema.\n\nEdit anyway?")) return;
          ta.readOnly = false;
          ta.focus();
          t.textContent = "Done Editing";
        } else {
          // Persist it. Toggling readOnly alone left the edit in the textarea
          // and nowhere else, so the change silently vanished on navigation.
          busy(t, true);
          call("save_prompt", { text: ta.value })
            .then(function (r) {
              busy(t, false);
              ta.readOnly = true;
              t.textContent = "Edit Prompt";
              toast("Prompt saved (" + r.chars + " characters). " +
                    "Restore Default is now available.", "ok");
            })
            .catch(function (e) { busy(t, false); toast(e.message, "danger"); });
        }
        return;
      }
    });
  }

  // -- theme ----------------------------------------------------------------
  //
  // Stored in localStorage rather than the settings table: it is a per-machine
  // display preference, it must apply before the first paint to avoid a flash
  // of the wrong theme, and waiting on a database round trip to decide what
  // colour the window is would do exactly that.

  const THEME_KEY = "saledeed.theme";

  function applyTheme(name) {
    document.documentElement.setAttribute("data-theme", name);
    const button = $("#btn-theme");
    if (button) {
      // The icon shows what the click will *give* you, not what you have.
      button.innerHTML = name === "dark" ? "&#9788;" : "&#9789;";
      button.title = name === "dark" ? "Switch to light theme" : "Switch to dark theme";
    }
  }

  function toggleTheme() {
    const next = document.documentElement.getAttribute("data-theme") === "dark"
      ? "light" : "dark";
    try { localStorage.setItem(THEME_KEY, next); } catch (e) { /* private mode */ }
    applyTheme(next);
  }

  function restoreTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (e) { /* ignore */ }
    applyTheme(saved === "dark" ? "dark" : "light");
  }

  // -- download ------------------------------------------------------------

  //: Where the next download is written. Empty means the default exports
  //: folder, which is what the original button always did.
  let downloadLocation = "";

  function downloadStatus(message, level) {
    const el = $("#download-status");
    if (!el) return;
    el.textContent = message;
    el.className = "hint" + (level ? " " + level : "");
  }

  function syncDownloadState() {
    const select = $("#download-batch");
    const button = $("#btn-download-batch");
    if (!select || !button) return;
    const chosen = !!select.value;
    button.disabled = !chosen;
    button.title = chosen ? "" : "Select a batch first";
    if (!chosen) {
      downloadStatus("Select a batch to begin.");
    } else if (downloadLocation) {
      downloadStatus("Ready to download to " + downloadLocation);
    } else {
      downloadStatus("Ready. Choose a location, or download to the default "
                     + "exports folder.");
    }
  }

  // -- failed-OCR selection -------------------------------------------------

  function selectedFailed() {
    return $$(".chk-failed:checked").map(function (c) { return +c.value; });
  }

  function syncFailedSelection() {
    const chosen = selectedFailed();
    const button = $("#btn-rerun-selected"), label = $("#selected-count");
    if (button) button.disabled = chosen.length === 0;
    if (label) {
      label.textContent = chosen.length
        ? chosen.length + " file(s) selected."
        : "No files selected.";
    }
    const all = $("#chk-all-failed"), boxes = $$(".chk-failed");
    if (all) {
      all.checked = boxes.length > 0 && chosen.length === boxes.length;
      // Neither all nor none: shown as a dash rather than a misleading tick.
      all.indeterminate = chosen.length > 0 && chosen.length < boxes.length;
    }
  }

  // Delegated, because the table is replaced wholesale on every refresh and a
  // listener bound to the current rows would stop working after the first one.
  document.addEventListener("change", function (event) {
    const t = event.target;
    if (!t) return;
    if (t.id === "chk-all-failed") {
      $$(".chk-failed").forEach(function (c) { c.checked = t.checked; });
      syncFailedSelection();
      return;
    }
    if (t.classList && t.classList.contains("chk-failed")) syncFailedSelection();
    if (t.id === "download-batch" || t.id === "download-scope") syncDownloadState();
  });

  function collectFilters() {
    const f = {};
    if ($("#q")) f.query = $("#q").value;
    if ($("#filter-batch")) f.batch_id = $("#filter-batch").value;
    if ($("#filter-status")) f.status = $("#filter-status").value;
    if ($("#sort")) f.sort = $("#sort").value;
    return f;
  }

  function submitBatch(button) {
    const username = ($("#username") || {}).value || "";
    const name = ($("#batch-name") || {}).value || "";
    if (!username.trim()) { toast("A username is required.", "warn"); return; }
    if (!name.trim()) { toast("A batch name is required.", "warn"); return; }
    busy(button, true);
    call("add_batch", { username: username, name: name })
      .then(function (r) {
        busy(button, false);
        toast("Batch <b>" + r.name + "</b> queued with " + r.files + " document(s).", "ok");
        location.hash = "#dashboard";
        navigate("dashboard");
      })
      .catch(function (e) { busy(button, false); toast(e.message, "danger"); });
  }

  function saveSettings(button) {
    const payload = {};
    [["devanagari-as", "translation_devanagari_as"],
     ["ocr-dpi", "ocr_dpi"], ["batch-mode", "batch_mode"], ["llm-mode", "llm_mode"],
     ["stamp-multiplier", "stamp_value_multiplier"], ["update-url", "update_repo_url"],
     ["prompt", "prompt"]].forEach(function (pair) {
      const el = $("#" + pair[0]);
      if (el) payload[pair[1]] = el.value;
    });
    if ($("#debug-logs")) payload.debug_logs = $("#debug-logs").checked ? "true" : "false";
    busy(button, true);
    call("save_settings", payload)
      .then(function () { busy(button, false); toast("Settings saved.", "ok"); })
      .catch(function (e) { busy(button, false); toast(e.message, "danger"); });
  }

  function saveRules(button) {
    const rules = {};
    $$("[data-rule]").forEach(function (el) { rules[el.dataset.rule] = el.checked; });
    const nums = {};
    [["pan-coverage", "pan_coverage_threshold"], ["pan-min-unmatched", "pan_coverage_min_unmatched"],
     ["pan-split", "pan_split_threshold"], ["proximity", "pan_aadhaar_proximity_chars"]]
      .forEach(function (p) { const el = $("#" + p[0]); if (el) nums[p[1]] = el.value; });
    busy(button, true);
    call("save_rules", { rules: rules, thresholds: nums })
      .then(function () { busy(button, false); toast("Validation rules saved.", "ok"); })
      .catch(function (e) { busy(button, false); toast(e.message, "danger"); });
  }

  // ----------------------------------------------------------------- modal

  function openModal(template, params) {
    call("fragment", { template: template, params: params || {} })
      .then(function (r) {
        const host = $("#modal-host");
        if (!host || !r.html) return;
        host.innerHTML = r.html;
        const backdrop = $("[data-modal-backdrop]", host);
        if (backdrop) {
          backdrop.addEventListener("click", function (ev) {
            if (ev.target === backdrop) closeModal();
          });
        }
      })
      .catch(function (e) { toast(e.message, "danger"); });
  }

  function legacyCopy(text) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing to offer */ }
    area.remove();
  }

  function closeModal() {
    const host = $("#modal-host");
    if (host) host.innerHTML = "";
  }

  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") closeModal();
  });

  // ------------------------------------------------------------- dropzone

  function bindDropzone() {
    $$(".dropzone").forEach(function (zone) {
      ["dragenter", "dragover"].forEach(function (evt) {
        zone.addEventListener(evt, function (e) {
          e.preventDefault(); zone.classList.add("over");
        });
      });
      ["dragleave", "drop"].forEach(function (evt) {
        zone.addEventListener(evt, function (e) {
          e.preventDefault(); zone.classList.remove("over");
        });
      });
      zone.addEventListener("drop", function (e) {
        const paths = [];
        if (e.dataTransfer && e.dataTransfer.files) {
          for (let i = 0; i < e.dataTransfer.files.length; i++) {
            const f = e.dataTransfer.files[i];
            // Qt exposes a real filesystem path; a browser will not.
            if (f.path) paths.push(f.path);
          }
        }
        if (!paths.length) { toast("Use Browse Files to select PDFs.", "warn"); return; }
        call("add_files", { paths: paths })
          .then(function () { navigate(currentPage); })
          .catch(function (err) { toast(err.message, "danger"); });
      });
    });
  }

  // ------------------------------------------------------------------ boot

  window.addEventListener("hashchange", function () {
    const p = pageFromHash();
    if (p !== currentPage) navigate(p);
  });

  // Applied before anything else so the window never flashes the wrong theme.
  restoreTheme();

  document.addEventListener("DOMContentLoaded", function () {
    currentPage = pageFromHash();
    restoreTheme();
    syncTabs(currentPage);
    bindGlobal();
    bindDropzone();
    connect().then(function (bridge) {
      if (bridge) startPolling();
    });
  });
})();
