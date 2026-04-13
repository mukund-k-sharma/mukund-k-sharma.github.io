document.addEventListener("DOMContentLoaded", () => {
    
    // 1. Mouse Spotlight Effect (with safety check)
    const cursorGlow = document.getElementById('cursor-glow');
    if (cursorGlow) {
        document.body.addEventListener('mousemove', (e) => {
            cursorGlow.style.background = `radial-gradient(600px circle at ${e.clientX}px ${e.clientY}px, rgba(45, 212, 191, 0.04), transparent 80%)`;
        });
    }

    // 2. Scroll Spy (Highlights the current section in the sidebar nav)
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');

    // Only run if we actually have sections and links to observe
    if (sections.length > 0 && navLinks.length > 0) {
        const observerOptions = {
            root: null,
            rootMargin: '0px',
            threshold: 0.3 // Changed to 0.3 to trigger slightly earlier
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    // Remove active class from all links
                    navLinks.forEach(link => link.classList.remove('active'));
                    
                    // Find the link that matches the intersecting section's ID
                    const activeId = entry.target.getAttribute('id');
                    const activeLink = document.querySelector(`.nav-link[href="#${activeId}"]`);
                    
                    if (activeLink) {
                        activeLink.classList.add('active');
                    }
                }
            });
        }, observerOptions);

        // Observe all sections
        sections.forEach(section => {
            observer.observe(section);
        });
    }

    // 3. Smooth Scrolling for Sidebar Links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // 4. Fallback Reveal Animation (If you still had old .reveal elements in HTML)
    function reveal() {
        var reveals = document.querySelectorAll(".reveal");
        for (var i = 0; i < reveals.length; i++) {
            var windowHeight = window.innerHeight;
            var elementTop = reveals[i].getBoundingClientRect().top;
            var elementVisible = 150;
            if (elementTop < windowHeight - elementVisible) {
                reveals[i].classList.add("active");
            }
        }
    }
    window.addEventListener("scroll", reveal);
    reveal(); // Trigger on load
});