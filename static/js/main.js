/* SafeExplorer project page: theme toggle, clip playback, BibTeX copy. Loaded in <head> so the theme applies before paint. */
(function () {
  "use strict";
  var KEY = "safeexplorer-theme";
  var root = document.documentElement;

  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function store(v) { try { localStorage.setItem(KEY, v); } catch (e) { /* storage unavailable */ } }
  var fromUrl = new URLSearchParams(location.search).get("theme");
  var initial = fromUrl === "dark" || fromUrl === "light" ? fromUrl : stored();
  if (initial === "dark" || initial === "light") root.setAttribute("data-theme", initial);

  function current() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var toggle = document.querySelector(".theme");
    toggle.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store(next);
    });

    var clips = Array.prototype.slice.call(document.querySelectorAll("video[data-autoplay]"));
    clips.forEach(function (v) {
      v.addEventListener("mouseenter", function () { v.controls = true; });
      v.addEventListener("mouseleave", function () { if (document.activeElement !== v) v.controls = false; });
      v.addEventListener("focus", function () { v.controls = true; });
      v.addEventListener("blur", function () { v.controls = false; });
    });
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { var p = e.target.play(); if (p) p.catch(function () {}); } else { e.target.pause(); }
        });
      }, { threshold: 0.4 });
      clips.forEach(function (v) { io.observe(v); });
    }

    /* safe-region sketch: the slide animation, redrawn in SVG so it follows the page theme */
    var sr = document.querySelector(".sr-anim");
    if (sr && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      var N = +sr.dataset.frames, CX = 278, CY = 238, STEP = 90;
      var region = sr.querySelector(".sr-region"), rings = sr.querySelectorAll(".sr-ring");
      var dots = sr.querySelectorAll(".sr-dot"), arrow = sr.querySelector(".sr-arrow"), recLabel = sr.querySelector(".sr-rec-label");
      var frameAt = function (f) {
        f = ((f % N) + N) % N;
        var u = f / (N - 1), radius = 60 + 95 * Math.min(1, u / 0.7);
        var theta = 4 * Math.PI * f / N, rho = radius * (0.5 + 0.68 * Math.abs(Math.sin(3 * Math.PI * f / N)));
        return { radius: radius, x: CX + rho * Math.cos(theta), y: CY + rho * Math.sin(theta), out: rho > radius };
      };
      var draw = function (f) {
        var now = frameAt(f);
        region.setAttribute("r", now.radius.toFixed(1));
        [60, 92, 124].forEach(function (r, i) { rings[i].toggleAttribute("hidden", !(r < now.radius - 6)); });
        for (var k = 0; k < dots.length; k++) {
          var g = f - dots.length + 1 + k;
          dots[k].toggleAttribute("hidden", g < 0);
          var p = frameAt(g);
          dots[k].setAttribute("cx", p.x.toFixed(1));
          dots[k].setAttribute("cy", p.y.toFixed(1));
          dots[k].classList.toggle("out", p.out);
        }
        var norm = Math.hypot(CX - now.x, CY - now.y);
        arrow.setAttribute("x1", now.x.toFixed(1)); arrow.setAttribute("y1", now.y.toFixed(1));
        arrow.setAttribute("x2", (now.x + 34 * (CX - now.x) / norm).toFixed(1));
        arrow.setAttribute("y2", (now.y + 34 * (CY - now.y) / norm).toFixed(1));
        recLabel.setAttribute("x", (now.x + 12).toFixed(1)); recLabel.setAttribute("y", (now.y - 12).toFixed(1));
        arrow.toggleAttribute("hidden", !now.out); recLabel.toggleAttribute("hidden", !now.out);
      };
      var HOLD = 12, frame = 0, timer = null;
      var start = function () { if (!timer) timer = setInterval(function () { frame = (frame + 1) % (N + HOLD); draw(Math.min(frame, N - 1)); }, STEP); };
      var stop = function () { clearInterval(timer); timer = null; };
      draw(frame);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (e) { e[0].isIntersecting ? start() : stop(); }).observe(sr);
      } else { start(); }
    }

    var copy = document.querySelector(".copy");
    copy.addEventListener("click", function () {
      var text = document.getElementById("bibtex").textContent;
      navigator.clipboard.writeText(text).then(function () {
        copy.textContent = "Copied";
        setTimeout(function () { copy.textContent = "Copy"; }, 1600);
      }, function () {
        var range = document.createRange();
        range.selectNodeContents(document.getElementById("bibtex"));
        getSelection().removeAllRanges();
        getSelection().addRange(range);
      });
    });
  });
})();
