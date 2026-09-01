/// Times any expression and prints "[timing] <label>: <elapsed>" to stdout,
/// then returns the expression's value unchanged — so it composes with `?`,
/// assignment, anything.
#[macro_export]
macro_rules! timed {
    ($label:expr, $block:expr) => {{
        let start = std::time::Instant::now();
        let result = $block;
        println!("[timing] {}: {:.3?}", $label, start.elapsed());
        result
    }};
}
