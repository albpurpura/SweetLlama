import Foundation
import llama

public enum LlmError: Error {
    case failedToLoadModel
    case failedToCreateContext
    case failedToCreateSampler
    case notLoaded
    case batchSizeExceedsTokenCount
}

public class LLM {
    private var params: CommonParams = .init()
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var smpl: UnsafeMutablePointer<llama_sampler>?
    private var nCTX: Int = 0
    
    private var isInterrupted: Bool = false
    private var hasNextToken: Bool = false
    private var generatedText: String = ""
    
    private var nPromptTokens: Int = 0
    private var nPast: Int = 0
    private var nRemaining: Int = 0
    
    private var embd: [llama_token] = []
    
    private var truncated: Bool = false
    private var stoppedEos: Bool = false
    private var stoppedWord: Bool = false
    private var stoppedLimit: Bool = false
    private var stoppingWord: String = ""
    private var modelPath: String = ""
    
    public var predictionFinished: Bool {
        !hasNextToken
    }
    
    public var generated: String {
        generatedText
    }
    
    public init() {
        llama_backend_init()
    }
    
    deinit {
        unload()
        llama_backend_free()
    }
    
    /// Loads the model from the given path, creates a context, and initializes the sampler.
    /// - Parameters:
    ///  - modelPath: The path to the model file.
    ///  - params: Parameters for model and sampling.
    ///  - Throws: An error if the model fails to load, the context fails to create, or the sampler fails to initialize.
    public func load(modelPath: String, params: CommonParams, isEmbeddingModel: Bool, embeddingBatchSize: Int) throws {
        unload()
        self.modelPath = modelPath
        self.params = params
        let initResult = LlamaCommon.initFrom(modelPath, params, isEmbeddingModel: isEmbeddingModel, embeddingBatchSize: embeddingBatchSize)
        model = initResult.model
        ctx = initResult.ctx
        guard let model = model else {
            throw LlmError.failedToLoadModel
        }
        guard let ctx = ctx else {
            throw LlmError.failedToCreateContext
        }
        nCTX = Int(llama_n_ctx(ctx))
        
        if !isEmbeddingModel {
            // embedding models don't need a sampler
            smpl = LlamaCommon.samplerInit(
                model, params.sparams, seed: UInt32.random(in: 0..<UInt32.max))
            guard let _ = smpl else {
                throw LlmError.failedToCreateSampler
            }
        }
    }
    
    /// Unloads the model, context, and sampler.
    /// This method has no effect if the model has not been loaded.
    public func unload() {
        if let ctx = ctx {
            llama_free(ctx)
        }
        if let model = model {
            llama_free_model(model)
        }
        if self.modelPath.contains("Llama") {
            if let smpl = smpl {
                llama_sampler_free(smpl)
            }
        }
    }
    
    /// Accepts a prompt and prepares for prediction.
    public func acceptPrompt(_ prompt: String) throws {
        guard let model = model, let ctx = ctx, let smpl = smpl else {
            throw LlmError.notLoaded
        }
        
        var promptTokens = LlamaCommon.tokenize(model, prompt, true, true)
        nPromptTokens = promptTokens.count
        
        if params.nKeep < 0 {
            params.nKeep = nPromptTokens
        }
        params.nKeep = min(nCTX - 4, params.nKeep)
        
        if nPromptTokens >= nCTX {
            truncatePrompt(&promptTokens)
            nPromptTokens = promptTokens.count
        }
        
        for token in promptTokens {
            llama_sampler_accept(smpl, token)
        }
        
        nPast = matchingPrefix(embd, promptTokens)
        
        embd = promptTokens
        if nPast == nPromptTokens {
            // eval at least 1 token to generate logits
            nPast -= 1
        }
        
        llama_kv_cache_seq_rm(ctx, 0, Int32(nPast), Int32(-1))
        
        llama_perf_context_reset(ctx)
        
        hasNextToken = true
        nRemaining = params.nPredict
        
        isInterrupted = false
        generatedText = ""
        
        truncated = false
        stoppedEos = false
        stoppedWord = false
        stoppedLimit = false
        stoppingWord = ""
    }
    
    func llama_batch_add(_ batch: inout llama_batch, _ id: llama_token, _ pos: llama_pos, _ seq_ids: [llama_seq_id], _ logits: Bool) {
        batch.token   [Int(batch.n_tokens)] = id
        batch.pos     [Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits  [Int(batch.n_tokens)] = logits ? 1 : 0
        
        batch.n_tokens += 1
    }
    
    // Helper function to match C++ batch_add_seq for embedding purposes
    func batchAddSeq(batch: inout llama_batch, tokens: [Int32], seqId: Int32) throws {
        let nTokens = tokens.count
        for i in 0..<nTokens {
            llama_batch_add(&batch, tokens[i], Int32(i), [seqId], true)
        }
    }
    
    public func encodeText(text: String) throws -> [Double] {
        guard let model = model, let ctx = ctx else {
            throw LlmError.notLoaded
        }
        
        let nEmbd = Int(llama_n_embd(model))
        // batch size of the model based on model card https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
        let nBatch = 8192
        
        // Tokenize input text
        var tokens = LlamaCommon.tokenize(model, text, true, true)
        let nTokens = tokens.count
        
        // Ensure tokens do not exceed batch size
        if nTokens > nBatch {
            print("Batch size exceeds token count.")
            throw LlmError.batchSizeExceedsTokenCount
        }
        
        // Prepare the batch
        var batch = llama_batch_init(Int32(nBatch), 0, 1)
        
        // Add all tokens as a single sequence using our helper function
        try batchAddSeq(batch: &batch, tokens: tokens, seqId: 0)
        
        // Clear KV cache
        llama_kv_cache_clear(ctx)
        
        // Run inference based on model type
        if llama_model_has_encoder(model) && !llama_model_has_decoder(model) {
            // Encoder-only model
            if llama_encode(ctx, batch) != 0 {
                throw LlmError.failedToCreateContext
            }
        } else if !llama_model_has_encoder(model) && llama_model_has_decoder(model) {
            // Decoder-only model
            if llama_decode(ctx, batch) != 0 {
                throw LlmError.failedToCreateContext
            }
        } else {
            print("Unsupported model type.")
            return []
        }
        
        // Fetch pooling type to decide how to fetch embeddings
        let poolingType = llama_pooling_type(ctx)
        
        // Extract embeddings based on pooling type
        var output = [Float](repeating: 0, count: nEmbd)  // Only need space for one embedding
        
        if poolingType == LLAMA_POOLING_TYPE_NONE {
            // Token embeddings
            for i in 0..<nTokens {
                guard let emb = llama_get_embeddings_ith(ctx, Int32(i)) else {
                    print("failed to extract token embeddings")
                    return []
                }
                // For token embeddings, we might want to do some form of pooling here
                for j in 0..<nEmbd {
                    output[j] += emb[j] / Float(nTokens) // Simple mean pooling
                }
            }
        } else {
            // Sequence embeddings - get single embedding for the whole sequence
            guard let emb = llama_get_embeddings_seq(ctx, 0) else {
                print("failed to extract sequence embeddings")
                return []
            }
            
            // Copy the sequence embedding
            for j in 0..<nEmbd {
                output[j] = emb[j]
            }
        }
        
        // Convert Float array to Double array
        return output.map { Double($0) }
    }
   
    /// Generates the next token.
    public func predict() throws -> String {
        guard let ctx = ctx else {
            throw LlmError.notLoaded
        }
        
        let token = nextToken()
        guard token >= 0 else {
            return ""
        }
        
        let tokenText = LlamaCommon.tokenToPiece(ctx, token, false)
        
        let str = String(cString: tokenText + [0], encoding: .utf8) ?? ""
        generatedText += str
        
        if !hasNextToken && nRemaining == 0 {
            stoppedLimit = true
        }
        
        return str
    }
    
    /// Marks prediction as finished.
    public func interrupt() {
        isInterrupted = true
    }
    
    private func nextToken() -> llama_token {
        guard let model = model, let ctx = ctx, let smpl = smpl else {
            return -1
        }
        
        var result: llama_token = -1
        
        if embd.count >= params.nCTX {
            let nLeft = nPast - params.nKeep
            let nDiscard = nLeft / 2
            
            llama_kv_cache_seq_rm(
                ctx, 0, Int32(params.nKeep + 1),
                Int32(params.nKeep + nDiscard + 1))
            llama_kv_cache_seq_add(
                ctx, 0, Int32(params.nKeep + 1 + nDiscard), Int32(nPast),
                Int32(-nDiscard))
            
            for i in params.nKeep + 1 + nDiscard..<embd.count {
                embd[i - nDiscard] = embd[i]
            }
            embd.removeLast(nDiscard)
            
            nPast -= nDiscard
        }
        
        while nPast < embd.count {
            var nEval = embd.count - nPast
            if nEval > params.nBatch {
                nEval = params.nBatch
            }
            
            if embd.withUnsafeMutableBufferPointer({ buffer in
                llama_decode(
                    ctx,
                    llama_batch_get_one(
                        buffer.baseAddress! + nPast, Int32(nEval)))
            }) != 0 {
                hasNextToken = false
                return result
            }
            nPast += nEval
            
            if isInterrupted {
                embd.removeLast(embd.count - nPast)
                hasNextToken = false
                return result
            }
        }
        
        if params.nPredict == 0 {
            hasNextToken = false
            result = llama_token_eos(model)
            return result
        }
        
        result = llama_sampler_sample(smpl, ctx, -1)
        
        embd.append(result)
        nRemaining -= 1
        
        if !embd.isEmpty && llama_token_is_eog(model, result) {
            hasNextToken = false
            stoppedEos = true
            return result
        }
        
        hasNextToken = params.nPredict == -1 || nRemaining != 0
        return result
    }
    
    private func truncatePrompt(_ promptTokens: inout [llama_token]) {
        let nLeft = nCTX - params.nKeep
        let nBlockSize = nLeft / 2
        let erasedBlocks =
        (promptTokens.count - params.nKeep - nBlockSize) / nBlockSize
        
        var newTokens: [llama_token] = []
        newTokens.append(contentsOf: promptTokens[0..<params.nKeep])
        newTokens.append(
            contentsOf: promptTokens[
                (params.nKeep + erasedBlocks * nBlockSize)...])
        
        print(
            "input truncated, n_ctx: \(nCTX), n_keep: \(params.nKeep), n_left: \(nLeft)"
        )
        
        truncated = true
        promptTokens = newTokens
    }
    
    private func matchingPrefix(_ a: [llama_token], _ b: [llama_token]) -> Int {
        var i = 0
        while i < a.count && i < b.count && a[i] == b[i] {
            i += 1
        }
        return i
    }
}
