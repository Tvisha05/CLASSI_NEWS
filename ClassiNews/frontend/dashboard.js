/**
 * Dashboard JavaScript for ClassiNews
 * Handles article management, filtering, and API calls
 */

const API_BASE = 'http://localhost:8002';
const API_URL = `${API_BASE}/predict`;
const SAMPLE_API_URL = `${API_BASE}/articles/sample`;
const HEALTH_CHECK_URL = `${API_BASE}/health`;
const SAVE_API_URL = `${API_BASE}/articles/save`;
const SAVED_LIST_API_URL = `${API_BASE}/articles/saved`;
const DELETE_SAVED_URL = `${API_BASE}/articles/saved`;

let articles = [];

// Load saved articles from server on page load
async function loadSavedArticles() {
    try {
        const response = await fetch(SAVED_LIST_API_URL);
        if (response.ok) {
            return await response.json();
        }
    } catch (error) {
        console.error('Error loading saved articles:', error);
    }
    return [];
}

// Save article to server
async function saveArticleToServer(article) {
    try {
        await fetch(SAVE_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(article)
        });
    } catch (error) {
        console.error('Error saving article:', error);
        showError('Failed to save article to server');
    }
}

// Delete article from server
async function deleteArticleFromServer(articleId) {
    try {
        await fetch(`${DELETE_SAVED_URL}/${articleId}`, {
            method: 'DELETE'
        });
    } catch (error) {
        console.error('Error deleting article:', error);
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', async function () {
    try {
        // Check backend connection
        const isBackendAvailable = await checkBackendConnection();
        if (!isBackendAvailable) {
            showError('Backend server is not running. Please start the backend server first.');
            return;
        }

        // Load saved articles from server
        const savedArticles = await loadSavedArticles();
        if (savedArticles.length > 0) {
            articles = savedArticles;
            renderArticles();
            updateStats();
        }
    } catch (error) {
        showError(`Failed to initialize: ${error.message}`);
    }
});

/**
 * Check backend connection
 */
async function checkBackendConnection() {
    try {
        const response = await fetch(HEALTH_CHECK_URL);
        return response.ok;
    } catch (error) {
        console.error('Backend connection error:', error);
        return false;
    }
}

let pendingArticle = null;

/**
 * Add a new article and classify it (No title field anymore)
 */
async function addArticle() {
    const contentInput = document.getElementById('article-content');
    const addBtn = document.getElementById('classify-btn');
    const btnText = addBtn.querySelector('.btn-text');
    const loader = document.getElementById('loader');

    // Check backend connection first
    const isBackendAvailable = await checkBackendConnection();
    if (!isBackendAvailable) {
        showError('Backend server is not running. Please start the backend server first.');
        return;
    }

    const content = contentInput.value.trim();

    if (!content) {
        showError('Please enter article content to classify.');
        return;
    }

    // Show loading state
    addBtn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: content
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();

        // Auto-generate title from first 50 chars
        const autoTitle = content.substring(0, 50) + (content.length > 50 ? '...' : '');

        // Store pending article
        pendingArticle = {
            id: Date.now().toString(),
            title: autoTitle,
            text: content,
            category: data.category,
            confidence: data.confidence,
            timestamp: new Date().toISOString(),
            isSaved: false
        };

        // Show classification result
        showSuccessNotification(data.category, data.confidence);

        // Clear input
        contentInput.value = '';

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while classifying the article.');
    } finally {
        addBtn.disabled = false;
        btnText.style.display = 'inline';
        loader.style.display = 'none';
    }
}

/**
 * Load sample articles from API
 */
async function loadSampleArticles() {
    try {
        const response = await fetch(SAMPLE_API_URL);
        if (!response.ok) {
            throw new Error('Failed to load sample articles');
        }

        const data = await response.json();

        // Add sample articles to the list
        data.articles.forEach(article => {
            articles.push({
                id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
                title: article.text.substring(0, 50) + '...',
                text: article.text,
                category: article.category,
                confidence: article.confidence,
                timestamp: new Date().toISOString(),
                isSaved: false
            });
        });

        renderArticles();
        updateStats();
    } catch (error) {
        console.error('Error loading sample articles:', error);
        showError('Unable to load sample articles. Make sure the backend is running.');
    }
}

/**
 * Clear all articles
 */
async function clearArticles() {
    if (confirm('Are you sure you want to clear all articles?')) {
        // Delete all saved articles from server
        for (const article of articles) {
            if (article.isSaved) {
                await deleteArticleFromServer(article.id);
            }
        }
        articles = [];
        renderArticles();
        updateStats();
    }
}

/**
 * Filter articles by category
 */
function filterArticles() {
    renderArticles();
}

/**
 * Update statistics
 */
function updateStats() {
    // Update total articles count
    const totalStat = document.getElementById('total-articles-stat');
    if (totalStat) {
        totalStat.textContent = articles.length;
    }

    // Update category counts (match AG News-style labels from the model)
    const categoryCounts = {
        'World': 0,
        'Sports': 0,
        'Business': 0,
        'Sci/Tech': 0
    };

    articles.forEach(article => {
        if (categoryCounts[article.category] !== undefined) {
            categoryCounts[article.category]++;
        }
    });

    // Update DOM
    Object.keys(categoryCounts).forEach(category => {
        // ID format in HTML: count-world, count-sports, count-business, count-scitech
        const normalizedId =
            category === 'Sci/Tech' ? 'scitech' : category.toLowerCase();
        const countEl = document.getElementById(`count-${normalizedId}`);
        if (countEl) {
            countEl.textContent = categoryCounts[category];
        }
    });
}

/**
 * Render articles in the grid
 */
function renderArticles() {
    const grid = document.getElementById('articles-grid');
    const emptyState = document.getElementById('empty-state');
    const categoryFilter = document.getElementById('category-filter').value;
    const totalCount = document.getElementById('total-count');
    const filteredCount = document.getElementById('filtered-count');

    // Filter articles
    const filtered = categoryFilter === 'all'
        ? articles
        : articles.filter(a => a.category === categoryFilter);

    // Update stats
    if (totalCount) totalCount.textContent = articles.length;
    if (filteredCount) filteredCount.textContent = filtered.length;

    // Clear grid but preserve empty state
    const articlesOnly = Array.from(grid.children).filter(child => child.id !== 'empty-state');
    articlesOnly.forEach(child => child.remove());

    if (filtered.length === 0) {
        if (emptyState) {
            emptyState.style.display = 'block';
        }
        return;
    }

    if (emptyState) {
        emptyState.style.display = 'none';
    }

    // Create article cards
    filtered.forEach((article, index) => {
        const card = createArticleCard(article, index);
        grid.appendChild(card);
    });
}

/**
 * Create an article card element
 */
function createArticleCard(article, index) {
    const card = document.createElement('div');
    card.className = 'article-card';

    const categoryClass = article.category.toLowerCase();
    const snippet = article.text.length > 150 ? article.text.substring(0, 150) + '...' : article.text;

    card.innerHTML = `
        <div class="article-header">
            <span class="badge badge-${categoryClass}">${getCategoryIcon(article.category)} ${article.category}</span>
            <span style="color: var(--text-muted); font-size: 0.85rem; font-weight: 600;">${article.confidence}%</span>
        </div>
        <h3 class="article-title">${escapeHtml(article.title || 'Untitled Article')}</h3>
        <p class="article-snippet">${escapeHtml(snippet)}</p>
        <div class="article-footer">
            <span class="article-date">${formatTime(article.timestamp)}</span>
            <div class="article-actions">
                <button class="btn-delete" onclick="deleteArticle(event, ${index})" title="Delete article">🗑️</button>
            </div>
        </div>
    `;

    return card;
}

/**
 * Get category icon
 */
function getCategoryIcon(category) {
    const icons = {
        'World': '🌍',
        'Sports': '🏆',
        'Business': '💼',
        'Sci/Tech': '🔬'
    };
    return icons[category] || '📁';
}

/**
 * Delete an article
 */
async function deleteArticle(event, index) {
    event.stopPropagation();
    if (!confirm('Are you sure you want to delete this article?')) return;

    const categoryFilter = document.getElementById('category-filter').value;
    const filtered = categoryFilter === 'all'
        ? [...articles]
        : articles.filter(a => a.category === categoryFilter);

    const articleToDelete = filtered[index];

    if (articleToDelete.isSaved) {
        await deleteArticleFromServer(articleToDelete.id);
    }

    articles = articles.filter(a => a.id !== articleToDelete.id);
    renderArticles();
    updateStats();
}

/**
 * Show error message
 */
function showError(message) {
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');

    if (errorSection && errorMessage) {
        errorMessage.textContent = message;
        errorSection.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => {
            errorSection.style.display = 'none';
        }, 5000);
    }
}

/**
 * Format timestamp
 */
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show success notification with classification result
 */
function showSuccessNotification(category, confidence) {
    const modal = document.getElementById('success-modal');
    const modalCategory = document.getElementById('modal-category');
    const modalConfidence = document.getElementById('modal-confidence');

    if (modal && modalCategory && modalConfidence) {
        modalCategory.textContent = category;
        modalConfidence.textContent = confidence + '%';
        modal.style.display = 'flex';
    }
}

/**
 * Save article from modal and close
 */
async function saveAndCloseModal() {
    if (pendingArticle) {
        pendingArticle.isSaved = true;
        await saveArticleToServer(pendingArticle);
        articles.unshift(pendingArticle);
        renderArticles();
        updateStats();
        pendingArticle = null;
    }
    const modal = document.getElementById('success-modal');
    if (modal) modal.style.display = 'none';
}

/**
 * Close success modal (discards the article)
 */
function closeSuccessModal() {
    pendingArticle = null;
    const modal = document.getElementById('success-modal');
    if (modal) modal.style.display = 'none';
}

/**
 * Show About modal
 */
function showAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) modal.style.display = 'flex';
}

/**
 * Close About modal
 */
function closeAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) modal.style.display = 'none';
}
