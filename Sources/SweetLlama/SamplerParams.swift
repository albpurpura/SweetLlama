public class SamplerParams: Codable {
    public var temperature: Float = 0.2 // lowered temp from 0.7
    public var topK: Int = 40
    public var topP: Float = 0.95
    public var minP: Float = 0.05
    public var minKeep: Int = 0
    public var xtcThreshold: Float = 0.1
    public var xtcProbability: Float = 0.0
    public var typicalP: Float = 1.0
    public var penaltyLastN: Int = 64
    public var penaltyRepeat: Float = 1.5 // changed from 1.0
    public var penaltyFrequency: Float = 0.0
    public var penaltyPresent: Float = 0.0
    
    public var dryMultiplier: Float = 0.0
    public var dryBase: Float = 1.75
    public var dryAllowedLength: Int = 2
    public var dryPenaltyLastN: Int = -1
    
    public var mirostat: MirostatType = .none
    public var mirostatTau: Float = 5.0
    public var mirostatEta: Float = 0.1

    enum CodingKeys: CodingKey {
        case temperature, topK, topP, minP, minKeep, xtcThreshold, xtcProbability, typicalP,
             penaltyLastN, penaltyRepeat, penaltyFrequency, penaltyPresent,
             dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
             mirostat, mirostatTau, mirostatEta
    }
    
    public init() {}

    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        temperature = try container.decode(Float.self, forKey: .temperature)
        topK = try container.decode(Int.self, forKey: .topK)
        topP = try container.decode(Float.self, forKey: .topP)
        minP = try container.decode(Float.self, forKey: .minP)
        minKeep = try container.decode(Int.self, forKey: .minKeep)
        xtcThreshold = try container.decode(Float.self, forKey: .xtcThreshold)
        xtcProbability = try container.decode(Float.self, forKey: .xtcProbability)
        typicalP = try container.decode(Float.self, forKey: .typicalP)
        penaltyLastN = try container.decode(Int.self, forKey: .penaltyLastN)
        penaltyRepeat = try container.decode(Float.self, forKey: .penaltyRepeat)
        penaltyFrequency = try container.decode(Float.self, forKey: .penaltyFrequency)
        penaltyPresent = try container.decode(Float.self, forKey: .penaltyPresent)
        dryMultiplier = try container.decode(Float.self, forKey: .dryMultiplier)
        dryBase = try container.decode(Float.self, forKey: .dryBase)
        dryAllowedLength = try container.decode(Int.self, forKey: .dryAllowedLength)
        dryPenaltyLastN = try container.decode(Int.self, forKey: .dryPenaltyLastN)
        mirostat = try container.decode(MirostatType.self, forKey: .mirostat)
        mirostatTau = try container.decode(Float.self, forKey: .mirostatTau)
        mirostatEta = try container.decode(Float.self, forKey: .mirostatEta)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(temperature, forKey: .temperature)
        try container.encode(topK, forKey: .topK)
        try container.encode(topP, forKey: .topP)
        try container.encode(minP, forKey: .minP)
        try container.encode(minKeep, forKey: .minKeep)
        try container.encode(xtcThreshold, forKey: .xtcThreshold)
        try container.encode(xtcProbability, forKey: .xtcProbability)
        try container.encode(typicalP, forKey: .typicalP)
        try container.encode(penaltyLastN, forKey: .penaltyLastN)
        try container.encode(penaltyRepeat, forKey: .penaltyRepeat)
        try container.encode(penaltyFrequency, forKey: .penaltyFrequency)
        try container.encode(penaltyPresent, forKey: .penaltyPresent)
        try container.encode(dryMultiplier, forKey: .dryMultiplier)
        try container.encode(dryBase, forKey: .dryBase)
        try container.encode(dryAllowedLength, forKey: .dryAllowedLength)
        try container.encode(dryPenaltyLastN, forKey: .dryPenaltyLastN)
        try container.encode(mirostat, forKey: .mirostat)
        try container.encode(mirostatTau, forKey: .mirostatTau)
        try container.encode(mirostatEta, forKey: .mirostatEta)
    }
}

public enum MirostatType: Int, CaseIterable, Codable {
    case none
    case v1
    case v2
    
    public var description: String {
        switch self {
        case .none: "None"
        case .v1: "V1"
        case .v2: "V2"
        }
    }
}
