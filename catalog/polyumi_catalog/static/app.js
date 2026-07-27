// Highlight the clicked row within its column, clearing prior siblings.
// Selection state is view-only; it does not affect what the server returns.
document.body.addEventListener('click', (evt) => {
  const item = evt.target.closest('.item');
  if (!item) return;
  const body = item.closest('.col-body');
  if (!body) return;
  body.querySelectorAll('.item.selected').forEach((el) => el.classList.remove('selected'));
  item.classList.add('selected');
});

// When a column is swapped (new scenes/episodes/datasets loaded), clear stale
// downstream state visually isn't needed since those columns are re-rendered
// from scratch by the server on each request.

// Items are role="button"/tabindex="0" divs (htmx binds its hx-get to native click
// events), so Enter/Space need an explicit trigger for keyboard and screen-reader users.
document.body.addEventListener('keydown', (evt) => {
  if (evt.key !== 'Enter' && evt.key !== ' ') return;
  const item = evt.target.closest('.item');
  if (!item) return;
  evt.preventDefault();
  item.click();
});

// Mutation endpoints (e.g. MCAP export/Foxglove launch) return plain-text error
// bodies on failure; htmx won't swap non-2xx responses in by default, so surface
// them the simplest way available.
document.body.addEventListener('htmx:responseError', (evt) => {
  alert(evt.detail.xhr.responseText || 'Request failed.');
});

// 'x' toggles the currently-open episode's usable/unusable status. Looked up by id at
// keypress time (rather than bound once) since #detail-body — and the button inside it —
// is replaced wholesale on every htmx swap. Ignored while typing in a form field so it
// doesn't hijack the rename/assign/dataset-name inputs elsewhere on the page.
document.body.addEventListener('keydown', (evt) => {
  if (evt.key.toLowerCase() !== 'x' || evt.ctrlKey || evt.metaKey || evt.altKey) return;
  const tag = evt.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  const btn = document.getElementById('unusable-toggle-btn');
  if (!btn) return;
  evt.preventDefault();
  btn.click();
});
