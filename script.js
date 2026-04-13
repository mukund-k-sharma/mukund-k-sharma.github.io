// 1. Mouse Spotlight Effect
const cursorGlow = document.getElementById('cursor-glow');

document.body.addEventListener('mousemove', (e) => {
    // Updates the radial gradient to follow the mouse
    cursorGlow.style.background = `radial-gradient(600px circle at ${e.clientX}px ${e.clientY}px, rgba(45, 212, 191, 0.04), transparent 80%)`;
});

// 2. Scroll Spy (Highlights the current section in the sidebar nav)
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.nav-link');

const observerOptions = {
    root: null,
    rootMargin: '0px',
    threshold: 0.5 // Trigger when 50% of the section is visible
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