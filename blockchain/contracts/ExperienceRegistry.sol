// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Experience Registry for continual learning and knowledge sharing.
///         Stores simulation run metadata, IPFS pointers, and quality attestations.
///         Enables agents to learn from previous successful strategies.
contract ExperienceRegistry {
    /// @notice Simulation experience record
    struct Experience {
        bytes32 runHash;        // Unique hash of simulation parameters
        string ipfsCid;         // IPFS content ID with full simulation data
        bytes32 dataHash;       // Hash of simulation results for integrity
        uint256 score;          // Performance score (cells killed, efficiency, etc.)
        address submitter;      // Address that submitted the experience
        uint32 timestamp;       // Block timestamp
        uint16 attestations;    // Number of quality attestations received
        bool verified;          // Whether experience has been verified
    }
    
    /// @notice Strategy metadata for categorizing experiences
    struct StrategyMeta {
        string strategyType;    // e.g., "pheromone-guided", "LLM-queen", "hybrid"
        string modelUsed;       // LLM model identifier
        uint16 nanobotCount;    // Number of nanobots
        uint16 tumorRadius;     // Tumor size parameter
        bytes32 datasetHash;    // Hash of tumor geometry (BraTS subject)
    }
    
    /// @notice Attestation from a validator
    struct Attestation {
        address validator;
        uint32 timestamp;
        uint8 quality;          // Quality score 0-100
        string notes;           // Optional notes
    }
    
    // Events
    event ExperienceSubmitted(
        bytes32 indexed runHash,
        string ipfsCid,
        bytes32 dataHash,
        uint256 score,
        address indexed submitter
    );
    
    event ExperienceAttested(
        bytes32 indexed runHash,
        address indexed validator,
        uint8 quality
    );
    
    event ExperienceVerified(
        bytes32 indexed runHash,
        address indexed verifier
    );
    
    // Storage
    mapping(bytes32 => Experience) public experiences;
    mapping(bytes32 => StrategyMeta) public strategies;
    mapping(bytes32 => Attestation[]) public attestations;
    mapping(address => bool) public authorizedValidators;
    
    // Owner for managing validators
    address public owner;
    
    // Minimum attestations required for verification
    uint8 public minAttestations = 2;
    
    constructor() {
        owner = msg.sender;
        authorizedValidators[msg.sender] = true;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyValidator() {
        require(authorizedValidators[msg.sender], "Not authorized validator");
        _;
    }
    
    /// @notice Submit a new simulation experience
    function submitExperience(
        bytes32 runHash,
        string calldata ipfsCid,
        bytes32 dataHash,
        uint256 score,
        StrategyMeta calldata strategyMeta
    ) external {
        require(experiences[runHash].runHash == bytes32(0), "Experience already exists");
        require(score > 0, "Score must be positive");
        require(bytes(ipfsCid).length > 0, "IPFS CID required");
        
        experiences[runHash] = Experience({
            runHash: runHash,
            ipfsCid: ipfsCid,
            dataHash: dataHash,
            score: score,
            submitter: msg.sender,
            timestamp: uint32(block.timestamp),
            attestations: 0,
            verified: false
        });
        
        strategies[runHash] = strategyMeta;
        
        emit ExperienceSubmitted(runHash, ipfsCid, dataHash, score, msg.sender);
    }
    
    /// @notice Attest to the quality of an experience
    function attestExperience(
        bytes32 runHash,
        uint8 quality,
        string calldata notes
    ) external onlyValidator {
        require(experiences[runHash].runHash != bytes32(0), "Experience not found");
        require(quality <= 100, "Quality must be <= 100");
        require(!hasConfirmed(runHash, msg.sender), "Already attested");
        
        Attestation memory attestation = Attestation({
            validator: msg.sender,
            timestamp: uint32(block.timestamp),
            quality: quality,
            notes: notes
        });
        
        attestations[runHash].push(attestation);
        experiences[runHash].attestations++;
        
        emit ExperienceAttested(runHash, msg.sender, quality);
        
        // Auto-verify if enough attestations
        if (experiences[runHash].attestations >= minAttestations) {
            _verifyExperience(runHash);
        }
    }
    
    /// @notice Verify an experience (can be called manually if conditions met)
    function verifyExperience(bytes32 runHash) external onlyValidator {
        require(experiences[runHash].runHash != bytes32(0), "Experience not found");
        require(!experiences[runHash].verified, "Already verified");
        require(experiences[runHash].attestations >= minAttestations, "Insufficient attestations");
        
        _verifyExperience(runHash);
    }
    
    /// @notice Internal verification logic
    function _verifyExperience(bytes32 runHash) internal {
        experiences[runHash].verified = true;
        emit ExperienceVerified(runHash, msg.sender);
    }
    
    /// @notice Check if validator has already attested
    function hasConfirmed(bytes32 runHash, address validator) public view returns (bool) {
        Attestation[] storage atts = attestations[runHash];
        for (uint i = 0; i < atts.length; i++) {
            if (atts[i].validator == validator) {
                return true;
            }
        }
        return false;
    }
    
    /// @notice Get average quality score for an experience
    function getAverageQuality(bytes32 runHash) public view returns (uint8) {
        Attestation[] storage atts = attestations[runHash];
        if (atts.length == 0) return 0;
        
        uint256 total = 0;
        for (uint i = 0; i < atts.length; i++) {
            total += atts[i].quality;
        }
        return uint8(total / atts.length);
    }
    
    /// @notice Get all attestations for an experience
    function getAttestations(bytes32 runHash) external view returns (Attestation[] memory) {
        return attestations[runHash];
    }
    
    /// @notice Add a new validator
    function addValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = true;
    }
    
    /// @notice Remove a validator
    function removeValidator(address validator) external onlyOwner {
        authorizedValidators[validator] = false;
    }
    
    /// @notice Update minimum attestations required
    function setMinAttestations(uint8 newMin) external onlyOwner {
        require(newMin > 0, "Minimum must be positive");
        minAttestations = newMin;
    }
    
    /// @notice Get experience details
    function getExperience(bytes32 runHash) external view returns (Experience memory, StrategyMeta memory) {
        return (experiences[runHash], strategies[runHash]);
    }
}