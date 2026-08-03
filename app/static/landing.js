// Reveal observer for staggered scroll animations on landing page
(function initReveal() {
  // Bail if reduced motion is set
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return;
  }

  // Arm the reveal state on html element
  document.documentElement.classList.add('js-reveal');

  // Set up intersection observer for reveal animations
  if (!window.IntersectionObserver) {
    // Fallback: disable reveal state if IntersectionObserver is unavailable
    document.documentElement.classList.remove('js-reveal');
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-revealed');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -10%' }
  );

  document.querySelectorAll('[data-reveal]').forEach((el) => {
    observer.observe(el);
  });
})();
