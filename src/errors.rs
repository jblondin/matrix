error_chain! {
    errors {
        IndexError(s: &'static str) {
            description("Index error")
            display("Index error: {}", s)
        }
        DecompositionError(s: String) {
            description("Decomposition error")
            display("Decomposition error: {}", s)
        }
        SolveError(s: String) {
            description("Solver error")
            display("Solver error: {}", s)
        }
        CondError(s: String) {
            description("Condition error")
            display("Condition error: {}", s)
        }
    }
}
