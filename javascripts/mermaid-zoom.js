/* ==========================================================================
   Mermaid diagram zoom — dependency-free click-to-expand overlay.

   MkDocs Material renders each ```mermaid fence as an inline <svg> sized to the
   content column, which makes the larger architecture flowcharts/sequence
   diagrams hard to read. This adds an "expand" affordance to every rendered
   diagram: click it to open a fullscreen overlay where the diagram can be
   panned (drag) and zoomed (wheel / +- buttons / double-click), then dismissed
   with Esc, the close button, or a click on the backdrop.

   No external libraries: the pan/zoom is a plain CSS transform driven by a few
   pointer/wheel handlers, so it works offline and adds nothing to the bundle
   beyond this file.
   ========================================================================== */
(function () {
  "use strict";

  var enhanced = new WeakSet();
  var overlay = null;
  var stage = null;
  var view = null; // the cloned <svg> currently shown
  var state = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 };

  function buildOverlay() {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "mermaid-zoom-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Zoomable diagram");
    overlay.innerHTML =
      '<div class="mermaid-zoom-toolbar">' +
      '  <button type="button" data-act="out" aria-label="Zoom out">−</button>' +
      '  <button type="button" data-act="reset" aria-label="Reset zoom">Reset</button>' +
      '  <button type="button" data-act="in" aria-label="Zoom in">+</button>' +
      '  <button type="button" data-act="close" aria-label="Close" class="mermaid-zoom-close">✕</button>' +
      "</div>" +
      '<div class="mermaid-zoom-stage"></div>';
    stage = overlay.querySelector(".mermaid-zoom-stage");
    document.body.appendChild(overlay);

    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeOverlay();
    });
    overlay.querySelector('[data-act="close"]').addEventListener("click", closeOverlay);
    overlay.querySelector('[data-act="in"]').addEventListener("click", function () { zoomBy(1.25); });
    overlay.querySelector('[data-act="out"]').addEventListener("click", function () { zoomBy(0.8); });
    overlay.querySelector('[data-act="reset"]').addEventListener("click", resetView);

    stage.addEventListener("wheel", function (e) {
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? 1.12 : 0.89);
    }, { passive: false });
    stage.addEventListener("pointerdown", function (e) {
      state.dragging = true;
      state.startX = e.clientX - state.x;
      state.startY = e.clientY - state.y;
      stage.setPointerCapture(e.pointerId);
      stage.classList.add("is-grabbing");
    });
    stage.addEventListener("pointermove", function (e) {
      if (!state.dragging) return;
      state.x = e.clientX - state.startX;
      state.y = e.clientY - state.startY;
      applyTransform();
    });
    function endDrag() { state.dragging = false; stage.classList.remove("is-grabbing"); }
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("dblclick", function () { zoomBy(1.5); });

    document.addEventListener("keydown", function (e) {
      if (!overlay.classList.contains("is-open")) return;
      if (e.key === "Escape") closeOverlay();
    });
  }

  function applyTransform() {
    if (!view) return;
    view.style.transform =
      "translate(" + state.x + "px," + state.y + "px) scale(" + state.scale + ")";
  }
  function zoomBy(factor) {
    state.scale = Math.min(12, Math.max(0.2, state.scale * factor));
    applyTransform();
  }
  function resetView() {
    state.scale = 1; state.x = 0; state.y = 0;
    applyTransform();
  }

  function openOverlay(svg) {
    buildOverlay();
    stage.innerHTML = "";
    view = svg.cloneNode(true);
    view.removeAttribute("style");
    view.style.maxWidth = "none";
    view.style.maxHeight = "none";
    view.style.transformOrigin = "center center";
    view.style.willChange = "transform";
    stage.appendChild(view);
    resetView();
    overlay.classList.add("is-open");
    document.body.classList.add("mermaid-zoom-lock");
  }
  function closeOverlay() {
    if (!overlay) return;
    overlay.classList.remove("is-open");
    document.body.classList.remove("mermaid-zoom-lock");
    view = null;
  }

  function enhance(container) {
    if (enhanced.has(container)) return;
    var svg = container.querySelector("svg");
    if (!svg) return;
    enhanced.add(container);
    container.classList.add("mermaid-zoomable");

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mermaid-expand-btn";
    btn.setAttribute("aria-label", "Expand diagram to full screen");
    btn.title = "Expand diagram";
    btn.innerHTML =
      '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">' +
      '<path fill="currentColor" d="M9 3H3v6h2V5h4V3zm12 0h-6v2h4v4h2V3zM5 15H3v6h6v-2H5v-4zm16 0h-2v4h-4v2h6v-6z"/>' +
      "</svg>";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      openOverlay(svg);
    });
    container.appendChild(btn);
    // Clicking the diagram body also opens the overlay.
    svg.addEventListener("click", function () { openOverlay(svg); });
  }

  function scan(root) {
    var nodes = (root || document).querySelectorAll(".mermaid");
    nodes.forEach(function (container) {
      if (container.querySelector("svg")) {
        enhance(container);
      } else {
        // SVG not rendered yet — watch until Material injects it.
        var mo = new MutationObserver(function () {
          if (container.querySelector("svg")) {
            mo.disconnect();
            enhance(container);
          }
        });
        mo.observe(container, { childList: true, subtree: true });
      }
    });
  }

  // Material exposes document$ (an RxJS subject) that fires on every page load,
  // including instant-navigation swaps. Fall back to a plain load listener.
  if (typeof window.document$ !== "undefined" && window.document$.subscribe) {
    window.document$.subscribe(function () { scan(document); });
  } else if (document.readyState !== "loading") {
    scan(document);
  } else {
    document.addEventListener("DOMContentLoaded", function () { scan(document); });
  }
})();
