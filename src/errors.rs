error_chain! {
    errors {
        IndexError(s: &'static str) {
            description("Index error")
            display("Index error: {}", s)
        }
    }
}
