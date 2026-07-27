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
