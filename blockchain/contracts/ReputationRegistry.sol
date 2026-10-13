// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title ReputationRegistry (Antelligence v4.1)
/// @notice Integer-ppm author reputation. Rule (mirrored exactly by backend/chain/v4_1.py
///         `reputation_update`): start 1,000,000; contradicted -> r / 2 (floor);
///         not contradicted -> min(1,000,000, r + 100,000). Excluded from briefs when r < 300,000.
contract ReputationRegistry {
    uint32 public constant START_PPM = 1_000_000;
    uint32 public constant STEP_UP_PPM = 100_000;
    uint32 public constant EXCLUDE_BELOW_PPM = 300_000;

    /// @notice Only the deployer reports, so third parties cannot grief an author's score.
    address public immutable reporter;

    mapping(bytes32 => uint32) private score;
    mapping(bytes32 => bool) private seen;
    /// @dev keccak256(abi.encode(runId, author, claimId)) -> already applied. Makes report idempotent,
    ///      so a client retry or a replayed queue job cannot double-count a claim.
    mapping(bytes32 => bool) private reportedClaim;

    event Reported(bytes32 indexed runId, bytes32 indexed author, bytes32 claimId, bool contradicted, uint32 newPpm);
    event Excluded(bytes32 indexed author, uint32 ppm);

    error NotReporter();
    error DuplicateReport();

    constructor() {
        reporter = msg.sender;
    }

    function report(bytes32 runId, bytes32 author, bytes32 claimId, bool contradicted) external returns (uint32 newPpm) {
        if (msg.sender != reporter) revert NotReporter();
        bytes32 key = keccak256(abi.encode(runId, author, claimId));
        if (reportedClaim[key]) revert DuplicateReport();
        reportedClaim[key] = true;
        uint32 r = seen[author] ? score[author] : START_PPM;
        bool wasExcluded = r < EXCLUDE_BELOW_PPM;
        r = update(r, contradicted);
        seen[author] = true;
        score[author] = r;
        emit Reported(runId, author, claimId, contradicted, r);
        if (!wasExcluded && r < EXCLUDE_BELOW_PPM) emit Excluded(author, r);
        return r;
    }

    /// @notice The pure rule, exposed so the parity table can be asserted for any starting value.
    function update(uint32 r, bool contradicted) public pure returns (uint32) {
        if (contradicted) return r / 2;
        uint64 up = uint64(r) + STEP_UP_PPM;
        return up > START_PPM ? START_PPM : uint32(up);
    }

    function reported(bytes32 runId, bytes32 author, bytes32 claimId) external view returns (bool) {
        return reportedClaim[keccak256(abi.encode(runId, author, claimId))];
    }

    function reputationOf(bytes32 author) public view returns (uint32) {
        return seen[author] ? score[author] : START_PPM;
    }

    function isExcluded(bytes32 author) external view returns (bool) {
        return reputationOf(author) < EXCLUDE_BELOW_PPM;
    }
}
