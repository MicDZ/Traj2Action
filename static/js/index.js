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
})