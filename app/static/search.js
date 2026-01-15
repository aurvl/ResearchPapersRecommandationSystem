document.addEventListener('DOMContentLoaded', () => {
    setupSearch('search-desktop');
    setupSearch('search-mobile');
});

function setupSearch(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;

    // Create dropdown container
    // Ensure the parent has relative positioning
    if (getComputedStyle(input.parentElement).position === 'static') {
        input.parentElement.style.position = 'relative';
    }
    
    const dropdown = document.createElement('div');
    // Updated classes for better dark mode contrast
    dropdown.className = 'absolute left-0 right-0 mt-2 rounded-xl overflow-hidden hidden z-50 shadow-2xl max-h-96 overflow-y-auto bg-white/90 backdrop-blur-xl dark:bg-slate-900/95 dark:backdrop-blur-xl border border-gray-200 dark:border-gray-700';
    
    input.parentElement.appendChild(dropdown);

    let debounceTimer;

    input.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        clearTimeout(debounceTimer);
        
        if (query.length < 2) {
            dropdown.classList.add('hidden');
            dropdown.innerHTML = '';
            return;
        }

        debounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                if (!response.ok) throw new Error('Network response was not ok');
                const results = await response.json();
                
                renderResults(results, dropdown);
            } catch (error) {
                console.error('Search error:', error);
            }
        }, 300);
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });
    
    // Show dropdown again if input has value and is focused
    input.addEventListener('focus', () => {
        if (input.value.trim().length >= 2 && dropdown.children.length > 0) {
            dropdown.classList.remove('hidden');
        }
    });
}

function renderResults(results, dropdown) {
    dropdown.innerHTML = '';
    
    if (results.length === 0) {
        const noResult = document.createElement('div');
        noResult.className = 'p-4 text-sm text-gray-500 dark:text-gray-400 text-center';
        noResult.textContent = 'No results found';
        dropdown.appendChild(noResult);
    } else {
        results.forEach(article => {
            const item = document.createElement('a');
            item.href = `/article/${article.id}`;
            // Updated hover and text colors for dark mode
            item.className = 'block p-3 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors border-b border-gray-200/50 dark:border-gray-700/50 last:border-0 text-left';
            
            item.innerHTML = `
                <div class="font-semibold text-gray-900 dark:text-gray-100 text-sm line-clamp-1">${article.title}</div>
                <div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-1">${article.author || 'Unknown Author'}</div>
            `;
            
            dropdown.appendChild(item);
        });
    }
    
    dropdown.classList.remove('hidden');
}
