import Foundation
import llama

@LlmActor
public class LlmState {
    private let llm: LLM
    private var stopped: Bool = false

    public init(_ llm: LLM) {
        self.llm = llm
    }

    /// Creates an LlmState instance from the given arguments.
    /// - Parameters:
    ///  - modelPath: Path to the GGUF file.
    ///  - params: Parameters for loading and prediction.
    ///  - Throws: An error if the model fails to load, the context fails to create, or the sampler fails to initialize.
    public static func create(modelPath: String, params: CommonParams) throws
        -> LlmState
    {
        let llm = LLM()
        try llm.load(modelPath: modelPath, params: params)
        return LlmState(llm)
    }
    
    public func embedText(text: String) throws -> [Double]{
        do {
            var res = try llm.encodeText(text: text)
            return res
        } catch {
            print("Error embedding text: \(error)")
            return []
        }
        return []
    }
    
    /// Predicts the next words given a text prompt.
    /// - Parameters:
    ///  - text: The text prompt.
    ///  - cooldownMs: The cooldown in milliseconds between predictions.
    ///  - Returns: An async stream of predicted words.
    ///           The stream will yield predictions at a rate of `cooldownMs` milliseconds.
    ///           The stream will yield predictions until the model is done predicting or `stop()` is called.
    public func predict(text: String, cooldownMs: Int = 0)
        -> AsyncThrowingStream<String, Error>
    {
        stopped = false
        return .init { continuation in
            Task { @LlmActor in
                do {
                    try llm.acceptPrompt(text)
                    while !llm.predictionFinished && !stopped {
                        let result = try llm.predict()
                        continuation.yield(result)

                        // Let stopped be set if needed
                        await Task.yield()
                        
                        if cooldownMs > 0 {
                            try? await Task.sleep(
                                nanoseconds: UInt64(cooldownMs) * 1_000_000)
                        }
                    }

                    if stopped {
                        llm.interrupt()
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func stop() {
        stopped = true
    }
}
