window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 0;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  if (!image) {
    return;
  }
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
  var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
      carousels[i].on('before:show', function(){});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
      element.bulmaCarousel.on('before-show', function(state) {});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    var $interpSlider = $('#interpolation-slider');
    if ($interpSlider.length && NUM_INTERP_FRAMES > 0) {
      $interpSlider.on('input', function(event) {
        setInterpolationImage(this.value);
      });
      setInterpolationImage(0);
      $interpSlider.prop('max', NUM_INTERP_FRAMES - 1);
    }

  try { bulmaSlider.attach(); } catch (e) {}

    // Task 2 video tray toggle
    const task2Box = document.getElementById('task2-box');
    if (task2Box) {
  const video = document.getElementById('task2-video');
      const title = document.getElementById('task2-title');
      const label1 = document.getElementById('task2-tray-label');
      const label2 = document.getElementById('task2-tray-label-2');
      const yellowBtn = task2Box.querySelector('button[data-tray="yellow"]');
      const blueBtn = task2Box.querySelector('button[data-tray="blue"]');
      const yellowSrc = task2Box.getAttribute('data-yellow-src');
      const blueSrc = task2Box.getAttribute('data-blue-src');
      

      function setTray(tray) {
        let url = tray === 'blue' ? blueSrc : yellowSrc;
        // cache busting to force reload in Chrome if needed
        const sep = url.includes('?') ? '&' : '?';
        url = `${url}${sep}t=${Date.now()}`;
        try { video.pause(); } catch(e){}
        if (video) {
          try { video.removeAttribute('src'); } catch(e){}
          video.src = url;
          try { video.load(); } catch(e){}
          try { video.currentTime = 0; } catch(e){}
          const playPromise = video.play();
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {});
          }
        }
        const trayText = tray === 'blue' ? 'blue' : 'yellow';
        title.textContent = `Task 2: pick up the tomato and put it in the ${trayText} tray`;
        if (label1) label1.textContent = trayText;
        if (label2) label2.textContent = trayText;
        // button states
        [yellowBtn, blueBtn].forEach(btn => {
          btn.classList.remove('is-link');
          btn.setAttribute('aria-pressed', 'false');
        });
        const activeBtn = tray === 'blue' ? blueBtn : yellowBtn;
        activeBtn.classList.add('is-link');
        activeBtn.setAttribute('aria-pressed', 'true');
      }

  yellowBtn?.addEventListener('click', function(e){ e.stopPropagation(); setTray('yellow'); });
  blueBtn?.addEventListener('click', function(e){ e.stopPropagation(); setTray('blue'); });

      // default
      setTray('yellow');
    }
    
    // Comparison videos banner (fruits yellow/blue)
    const cmpVideo = document.getElementById('cmp-video');
    const cmpPrev = document.getElementById('cmp-prev');
    const cmpNext = document.getElementById('cmp-next');
    const cmpYellowBtn = document.getElementById('cmp-yellow');
    const cmpBlueBtn = document.getElementById('cmp-blue');
    const cmpCounter = document.getElementById('cmp-counter');
    const cmpCurrentLabel = document.getElementById('cmp-current-label');
  const cmpTaskTitle = document.getElementById('cmp-task-title');
  const cmpSubtitle = document.getElementById('cmp-subtitle');
    if (cmpVideo && cmpPrev && cmpNext && cmpYellowBtn && cmpBlueBtn && cmpCounter) {
      const yellowFiles = [
        'comparison_video_11_480p.mp4',
        'comparison_video_13_480p.mp4',
        'comparison_video_17_480p.mp4',
        'comparison_video_3_480p.mp4',
        'comparison_video_8_480p.mp4',
        'comparison_video_9_480p.mp4'
      ];
      const blueFiles = [
        'comparison_video_21_480p.mp4',
        'comparison_video_24_480p.mp4',
        'comparison_video_26_480p.mp4',
        'comparison_video_27_480p.mp4',
        'comparison_video_28_480p.mp4',
        'comparison_video_31_480p.mp4',
        'comparison_video_34_480p.mp4',
        'comparison_video_35_480p.mp4',
        'comparison_video_36_480p.mp4'
      ];
      let tray = 'yellow';
      let idx = 0;

      function files() { return tray === 'blue' ? blueFiles : yellowFiles; }
      function base() { return `static/videos/comparison/fruits_compare_480/${tray}`; }

      function updateVideo() {
        const list = files();
        if (!list.length) return;
        idx = (idx + list.length) % list.length; // guard bounds
        const src = `${base()}/${list[idx]}`;
        try { cmpVideo.pause(); } catch(e){}
        // cache busting to ensure reload
        const sep = src.includes('?') ? '&' : '?';
        cmpVideo.src = `${src}${sep}t=${Date.now()}`;
        try { cmpVideo.load(); } catch(e){}
        try { cmpVideo.currentTime = 0; } catch(e){}
        const p = cmpVideo.play();
        if (p && p.catch) p.catch(()=>{});
        cmpCounter.textContent = `${idx+1}/${list.length}`;
        if (cmpCurrentLabel) cmpCurrentLabel.textContent = tray;
        if (cmpTaskTitle) {
          const trayText = tray === 'blue' ? 'blue' : 'yellow';
          cmpTaskTitle.textContent = `pick up the tomato and put it in the ${trayText} tray`;
        }
        if (cmpSubtitle) {
          const trayText = tray === 'blue' ? 'blue' : 'yellow';
          cmpSubtitle.textContent = `pick up the tomato and put it in the ${trayText} tray`;
        }
      }

      function setTray(nextTray) {
        tray = nextTray;
        idx = 0;
        [cmpYellowBtn, cmpBlueBtn].forEach(btn => { btn.classList.remove('is-link'); btn.setAttribute('aria-pressed','false'); });
        const active = tray === 'blue' ? cmpBlueBtn : cmpYellowBtn;
        active.classList.add('is-link');
        active.setAttribute('aria-pressed','true');
        updateVideo();
      }

      cmpPrev.addEventListener('click', () => { idx -= 1; updateVideo(); });
      cmpNext.addEventListener('click', () => { idx += 1; updateVideo(); });
      cmpYellowBtn.addEventListener('click', () => setTray('yellow'));
      cmpBlueBtn.addEventListener('click', () => setTray('blue'));

      // Keyboard support (left/right)
      document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') { idx -= 1; updateVideo(); }
        else if (e.key === 'ArrowRight') { idx += 1; updateVideo(); }
      });

      // init
      setTray('yellow');
    }

    // Bottle comparison banner
    const btlVideo = document.getElementById('btl-video');
    const btlPrev = document.getElementById('btl-prev');
    const btlNext = document.getElementById('btl-next');
    const btlCounter = document.getElementById('btl-counter');
    if (btlVideo && btlPrev && btlNext && btlCounter) {
      const list = [
        'comparison_video_0_480p.mp4',
        'comparison_video_7_480p.mp4',
        'comparison_video_23_480p.mp4',
        'comparison_video_31_480p.mp4',
        'comparison_video_33_480p.mp4',
        'comparison_video_37_480p.mp4',
        'comparison_video_38_480p.mp4'
      ];
      let i = 0;
      const base = 'static/videos/comparison/bottle_compare_video';
      function update() {
        i = (i + list.length) % list.length;
        const src = `${base}/${list[i]}`;
        try { btlVideo.pause(); } catch(e){}
        const sep = src.includes('?') ? '&' : '?';
        btlVideo.src = `${src}${sep}t=${Date.now()}`;
        try { btlVideo.load(); } catch(e){}
        try { btlVideo.currentTime = 0; } catch(e){}
        const p = btlVideo.play(); if (p && p.catch) p.catch(()=>{});
        btlCounter.textContent = `${i+1}/${list.length}`;
      }
      btlPrev.addEventListener('click', ()=>{ i -= 1; update(); });
      btlNext.addEventListener('click', ()=>{ i += 1; update(); });
      // keyboard support when bottle section in view (basic)
      document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') { i -= 1; update(); }
        else if (e.key === 'ArrowRight') { i += 1; update(); }
      });
      update();
    }
})