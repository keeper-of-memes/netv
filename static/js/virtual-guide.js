/**
 * Virtual scrolling for the TV guide.
 * Only renders rows that are visible (plus buffer), fetches more as needed.
 */
class VirtualGuide {
  constructor(options) {
    this.container = options.container;
    this.rowHeight = options.rowHeight || 64; // 4rem = 64px desktop
    this.rowHeightMobile = options.rowHeightMobile || 40; // 2.5rem mobile
    this.totalRows = options.totalRows;
    this.bufferSize = options.bufferSize || 50;
    this.initialRows = options.initialRows || []; // SSR rows
    this.offset = options.offset || 0;
    this.cats = options.cats || '';
    this.logoUrlFilter = options.logoUrlFilter || (url => url);

    // State
    this.cache = new Map(); // row index -> row data
    this.renderedRange = { start: 0, end: 0 };
    this.pendingFetch = null;
    this.scrollDebounce = null;
    this.fetchDebounce = null;
    this.isMobile = window.innerWidth < 512;

    // DOM elements
    this.viewport = null;
    this.content = null;
    this.spacer = null;

    this.init();
  }

  get currentRowHeight() {
    return this.isMobile ? this.rowHeightMobile : this.rowHeight;
  }

  get visibleCount() {
    if (!this.viewport) return 30;
    return Math.ceil(this.viewport.clientHeight / this.currentRowHeight) + 1;
  }

  init() {
    // Cache initial SSR rows
    for (const row of this.initialRows) {
      this.cache.set(row.index, row);
    }

    // Set up virtual scroll container
    this.setupDOM();
    this.bindEvents();

    // Handle scroll position restoration
    // Check if there's a saved scroll position that's beyond initial rows
    const scrollKey = 'guide_scroll';
    const savedScroll = sessionStorage.getItem(scrollKey);

    if (savedScroll && this.viewport) {
      const scrollTop = parseInt(savedScroll);
      const firstVisible = Math.floor(scrollTop / this.currentRowHeight);

      // If saved position is beyond initial batch, fetch first then scroll
      if (firstVisible >= this.initialRows.length) {
        // Fetch data for the saved position, then restore scroll
        const start = Math.max(0, firstVisible - this.bufferSize);
        const end = Math.min(this.totalRows, firstVisible + this.visibleCount + this.bufferSize);

        this.fetchMissingRanges([{ start, end }]).then(() => {
          this.viewport.scrollTop = scrollTop;
          this.renderedRange = { start, end };
          this.render();
        });
        return; // Don't do normal init flow
      }
    }

    // If we have more rows than initial batch, enable virtual scrolling
    if (this.totalRows > this.initialRows.length) {
      this.updateVisibleRange();
    }
  }

  setupDOM() {
    // Find the scroll container (the overflow-y-auto div)
    this.viewport = this.container.querySelector('.overflow-y-auto');
    if (!this.viewport) return;

    // Create spacer for full height scrollbar
    this.spacer = document.createElement('div');
    this.spacer.className = 'virtual-spacer';
    this.spacer.style.height = `${this.totalRows * this.currentRowHeight}px`;
    this.spacer.style.position = 'absolute';
    this.spacer.style.top = '0';
    this.spacer.style.left = '0';
    this.spacer.style.right = '0';
    this.spacer.style.pointerEvents = 'none';

    // Create content container
    this.content = document.createElement('div');
    this.content.className = 'virtual-content';
    this.content.style.position = 'relative';
    this.content.style.zIndex = '1';

    // Move existing rows into content container
    const existingRows = this.viewport.querySelectorAll('.guide-row');
    existingRows.forEach(row => this.content.appendChild(row));

    // Set viewport to relative positioning
    this.viewport.style.position = 'relative';

    // Add spacer and content to viewport
    this.viewport.appendChild(this.spacer);
    this.viewport.insertBefore(this.content, this.spacer);

    // Set initial rendered range based on SSR content
    this.renderedRange = { start: 0, end: this.initialRows.length };
  }

  bindEvents() {
    if (!this.viewport) return;

    // Scroll handler with RAF for smooth updates
    let ticking = false;
    this.viewport.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          this.onScroll();
          ticking = false;
        });
        ticking = true;
      }
    }, { passive: true });

    // Handle resize
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        const wasMobile = this.isMobile;
        this.isMobile = window.innerWidth < 512;
        if (wasMobile !== this.isMobile) {
          // Row height changed, update spacer
          this.spacer.style.height = `${this.totalRows * this.currentRowHeight}px`;
          this.updateVisibleRange();
        }
      }, 100);
    });
  }

  onScroll() {
    // Clear any pending debounce
    clearTimeout(this.fetchDebounce);
    clearTimeout(this.renderDebounce);

    const scrollTop = this.viewport.scrollTop;
    const firstVisible = Math.floor(scrollTop / this.currentRowHeight);
    const lastVisible = firstVisible + this.visibleCount;

    // Track scroll direction
    const scrollDelta = scrollTop - (this.lastScrollTop || 0);
    this.lastScrollTop = scrollTop;
    if (Math.abs(scrollDelta) > 5) { // Ignore tiny movements
      this.scrollDirection = scrollDelta > 0 ? 'down' : 'up';
    }

    // Calculate desired range with buffer
    const desiredStart = Math.max(0, firstVisible - this.bufferSize);
    const desiredEnd = Math.min(this.totalRows, lastVisible + this.bufferSize);

    // Check if we need to update rendered range
    const needsRender = desiredStart < this.renderedRange.start ||
                        desiredEnd > this.renderedRange.end;

    if (needsRender) {
      // Render immediately with whatever we have (placeholders for missing)
      this.renderDebounce = setTimeout(() => {
        this.renderedRange = { start: desiredStart, end: desiredEnd };
        this.render();
      }, 16); // ~60fps for smooth visual update

      // Debounce fetching - wait for scroll to settle before fetching
      this.fetchDebounce = setTimeout(() => {
        this.updateVisibleRange();
      }, 150); // Wait 150ms for scroll to settle
    }
  }

  async updateVisibleRange() {
    const scrollTop = this.viewport.scrollTop;
    const firstVisible = Math.floor(scrollTop / this.currentRowHeight);
    const lastVisible = firstVisible + this.visibleCount;

    // Calculate ranges: visible, forward buffer, backward buffer
    const visibleStart = Math.max(0, firstVisible);
    const visibleEnd = Math.min(this.totalRows, lastVisible + 1);

    const bufferStart = Math.max(0, firstVisible - this.bufferSize);
    const bufferEnd = Math.min(this.totalRows, lastVisible + this.bufferSize);

    // Priority fetch order based on scroll direction
    const fetchOrder = [];

    // 1. Always fetch visible rows first
    const visibleMissing = this.findMissingRanges(visibleStart, visibleEnd);
    if (visibleMissing.length > 0) {
      fetchOrder.push({ ranges: visibleMissing, priority: 'visible' });
    }

    // 2. Fetch buffer in scroll direction
    // 3. Fetch buffer in opposite direction
    if (this.scrollDirection === 'down') {
      const forwardMissing = this.findMissingRanges(visibleEnd, bufferEnd);
      const backwardMissing = this.findMissingRanges(bufferStart, visibleStart);
      if (forwardMissing.length > 0) fetchOrder.push({ ranges: forwardMissing, priority: 'forward' });
      if (backwardMissing.length > 0) fetchOrder.push({ ranges: backwardMissing, priority: 'backward' });
    } else {
      const backwardMissing = this.findMissingRanges(bufferStart, visibleStart);
      const forwardMissing = this.findMissingRanges(visibleEnd, bufferEnd);
      if (backwardMissing.length > 0) fetchOrder.push({ ranges: backwardMissing, priority: 'backward' });
      if (forwardMissing.length > 0) fetchOrder.push({ ranges: forwardMissing, priority: 'forward' });
    }

    // Fetch in priority order, re-rendering after each batch
    for (const batch of fetchOrder) {
      await this.fetchMissingRanges(batch.ranges);
      // Re-render after each batch so visible content appears first
      this.renderedRange = { start: bufferStart, end: bufferEnd };
      this.render();
    }

    // Final render
    this.renderedRange = { start: bufferStart, end: bufferEnd };
    this.render();
  }

  findMissingRanges(start, end) {
    const ranges = [];
    let rangeStart = null;

    for (let i = start; i < end; i++) {
      if (!this.cache.has(i)) {
        if (rangeStart === null) rangeStart = i;
      } else if (rangeStart !== null) {
        ranges.push({ start: rangeStart, end: i });
        rangeStart = null;
      }
    }

    if (rangeStart !== null) {
      ranges.push({ start: rangeStart, end });
    }

    return ranges;
  }

  async fetchMissingRanges(ranges) {
    // Early return if no ranges to fetch
    if (!ranges || ranges.length === 0) {
      return;
    }

    // Merge into a single request for simplicity
    const overallStart = Math.min(...ranges.map(r => r.start));
    const overallEnd = Math.max(...ranges.map(r => r.end));

    // If there's a pending fetch, check if it's for a relevant range
    if (this.pendingFetch && this.pendingFetchRange) {
      const p = this.pendingFetchRange;
      const overlaps = !(overallEnd < p.start || overallStart > p.end);

      if (overlaps) {
        // Pending fetch will give us some useful data, let it finish
        return;
      } else {
        // Pending fetch is for a completely different area - abort it
        this.pendingFetch.abort();
        this.pendingFetch = null;
        this.pendingFetchRange = null;
      }
    }

    const controller = new AbortController();
    this.pendingFetch = controller;
    this.pendingFetchRange = { start: overallStart, end: overallEnd };

    try {
      // Don't pass cats - server uses saved user filter
      // This keeps URLs short and avoids length limits
      const params = new URLSearchParams({
        start: overallStart,
        count: overallEnd - overallStart,
        offset: this.offset
      });

      const resp = await fetch(`/api/guide/rows?${params}`, {
        signal: controller.signal
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const data = await resp.json();

      // Cache the fetched rows
      for (const row of data.rows) {
        this.cache.set(row.index, row);
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error('Failed to fetch guide rows:', e);
      }
    } finally {
      if (this.pendingFetch === controller) {
        this.pendingFetch = null;
        this.pendingFetchRange = null;
      }
    }
  }

  render() {
    if (!this.content) return;

    const html = [];

    for (let i = this.renderedRange.start; i < this.renderedRange.end; i++) {
      const row = this.cache.get(i);
      if (row) {
        html.push(this.renderRow(row, i));
      } else {
        html.push(this.renderPlaceholder(i));
      }
    }

    // Position content at the right scroll offset
    this.content.style.transform = `translateY(${this.renderedRange.start * this.currentRowHeight}px)`;
    this.content.innerHTML = html.join('');
  }

  renderPlaceholder(index) {
    const height = this.currentRowHeight;
    const isMobile = this.isMobile;

    if (isMobile) {
      return `
        <div class="guide-row flex border-b border-gray-700 animate-pulse" data-row="${index}" style="height: ${height}px;">
          <div class="compact-only w-32 flex-shrink-0 p-1 flex items-center bg-gray-800 sticky left-0 z-10 border-r border-gray-700">
            <div class="h-3 bg-gray-600 rounded w-20"></div>
          </div>
          <div class="flex-1 relative ml-1">
            <div class="absolute inset-0.5 bg-gray-700 rounded"></div>
          </div>
        </div>
      `;
    }

    return `
      <div class="guide-row flex border-b border-gray-700 animate-pulse" data-row="${index}" style="height: ${height}px;">
        <div class="desktop-only w-36 lg:w-48 flex-shrink-0 p-1 flex items-center gap-2 bg-gray-800 sticky left-0 z-10 border-r border-gray-700">
          <div class="w-10 h-10 bg-gray-600 rounded"></div>
          <div class="h-4 bg-gray-600 rounded w-24"></div>
        </div>
        <div class="flex-1 relative ml-1">
          <div class="absolute top-1 bottom-1 left-0 right-1/3 bg-gray-700 rounded"></div>
        </div>
      </div>
    `;
  }

  renderRow(row, index) {
    const ch = row.channel;
    const iconUrl = ch.icon ? this.logoUrlFilter(ch.icon) : '';
    const height = this.currentRowHeight;

    // Escape HTML in text content
    const escapeHtml = (str) => {
      if (!str) return '';
      return str.replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
    };

    // Desktop programs
    let programsDesktop = '';
    if (row.programs && row.programs.length > 0) {
      programsDesktop = row.programs.map((prog, pIdx) => `
        <a href="/play/live/${ch.stream_id}"
           class="absolute top-1 bottom-1 bg-gray-700 hover:bg-gray-600 rounded px-2 py-1 overflow-hidden
                  focusable border-2 border-transparent focus:border-blue-500 focus:bg-blue-900/50"
           style="left: ${prog.left_pct}%; width: calc(${prog.width_pct}% - 4px);"
           tabindex="0" data-nav="epg" data-row="${index}" data-col="${pIdx}"
           title="${escapeHtml(prog.title)}&#10;${prog.start} - ${prog.end}&#10;${escapeHtml(prog.desc)}">
          <div class="text-sm font-medium truncate">${escapeHtml(prog.title)}</div>
          <div class="text-xs text-gray-400 truncate">${escapeHtml(prog.desc)}</div>
        </a>
      `).join('');
    } else {
      programsDesktop = `
        <div class="absolute inset-1 flex items-center px-2 text-gray-500 text-sm">
          No program info
        </div>
      `;
    }

    // Mobile programs
    let programsMobile = '';
    if (row.programs_mobile && row.programs_mobile.length > 0) {
      programsMobile = row.programs_mobile.map((prog, pIdx) => `
        <a href="/play/live/${ch.stream_id}"
           class="absolute top-0.5 bottom-0.5 bg-gray-700 hover:bg-gray-600 rounded px-1 overflow-hidden
                  focusable border border-gray-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 focus:ring-offset-gray-800"
           style="left: ${prog.left_pct}%; width: calc(${prog.width_pct}% - 4px);"
           tabindex="0" data-nav="epg" data-row="${index}" data-col="${pIdx}"
           title="${escapeHtml(prog.title)}&#10;${prog.start} - ${prog.end}&#10;${escapeHtml(prog.desc)}">
          <div class="text-[10px] font-medium truncate">${escapeHtml(prog.title)}</div>
        </a>
      `).join('');
    } else {
      programsMobile = `
        <div class="absolute inset-0.5 flex items-center px-1 text-gray-500 text-[10px]">
          No info
        </div>
      `;
    }

    return `
      <div class="guide-row flex border-b border-gray-700 hover:bg-gray-750" data-row="${index}">
        <!-- Mobile Channel Info -->
        <div class="compact-only w-32 flex-shrink-0 p-1 items-center bg-gray-800 sticky left-0 z-10 border-r border-gray-700">
          <a href="/play/live/${ch.stream_id}"
             class="text-xs line-clamp-2 hover:text-blue-400 focus:text-blue-400 focus:outline focus:outline-2 focus:outline-blue-500 focusable"
             tabindex="0" data-nav="epg" data-row="${index}" data-col="-1"
             title="${escapeHtml(ch.name)}">
            ${escapeHtml(ch.name)}
          </a>
        </div>

        <!-- Desktop Channel Info -->
        <div class="desktop-only w-36 lg:w-48 flex-shrink-0 p-1 items-center gap-1 bg-gray-800 sticky left-0 z-10 border-r border-gray-700">
          ${iconUrl ? `<img src="${iconUrl}" alt="" class="w-10 h-10 object-contain" onerror="this.style.display='none'">` : ''}
          <a href="/play/live/${ch.stream_id}"
             class="text-sm line-clamp-3 hover:text-blue-400 focus:text-blue-400 focus:outline focus:outline-2 focus:outline-blue-500 focusable"
             tabindex="0" data-nav="epg" data-row="${index}" data-col="-1"
             title="${escapeHtml(ch.name)}">
            ${escapeHtml(ch.name)}
          </a>
        </div>

        <!-- Mobile Programs (2-hour window) -->
        <div class="compact-only-block flex-1 relative h-10 ml-1">
          ${programsMobile}
        </div>

        <!-- Desktop Programs -->
        <div class="desktop-only-block flex-1 relative h-16 ml-1">
          ${programsDesktop}
        </div>
      </div>
    `;
  }
}

// Export for use
window.VirtualGuide = VirtualGuide;
