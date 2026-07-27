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
