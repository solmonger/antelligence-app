// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title AnchorRegistry (Antelligence v4.1)
/// @notice Append-only Merkle-root anchors for board runs (docs/v4.1/INTERFACES.md section 3).
///         Writes are deployer-only (anchor and anchorOnce): a third party could otherwise anchor a
///         runId first, fix itself as owner and lock the real run out. The run owner is therefore
///         always the deployer; the getter is kept for the verifier.
///         The chain stores (runId, size, root) directly; anyone recomputes the anchor digest
///         sha256(runId || uint64_be(size) || root) that OpenTimestamps and Rekor also commit to.
contract AnchorRegistry {
    struct Run {
        address owner;
        uint64 lastSize;
        bytes32 lastRoot;
        uint64 count;
    }

    /// @notice Deployer. Only it may call anchor and anchorOnce, so neither a runId nor the well-known
    ///         single-shot ids (keccak256("v4.1|prereg"), keccak256("v4.1|runplan")) can be squatted.
    address public immutable deployer;

    mapping(bytes32 => Run) private runs;
    mapping(bytes32 => mapping(uint64 => bytes32)) private rootAt_;
    mapping(bytes32 => bytes32) private onceDigest_;

    event Anchored(bytes32 indexed runId, uint64 size, bytes32 root, bytes32 prevRoot, address indexed owner);
    event AnchoredOnce(bytes32 indexed id, bytes32 digest);

    error SizeNotIncreasing();
    error PrevRootMismatch();
    error EmptyRoot();
    error AlreadyAnchored();
    error NotAnchored();
    error NotDeployer();
    error EmptyDigest();

    constructor() {
        deployer = msg.sender;
    }

    /// @notice Anchor the Merkle root of the first `size` entries of a run.
    function anchor(bytes32 runId, uint64 size, bytes32 root, bytes32 prevRoot) external {
        if (msg.sender != deployer) revert NotDeployer();
        Run storage r = runs[runId];
        if (r.owner == address(0)) r.owner = msg.sender; // first anchor records the owner (always the deployer)
        if (root == bytes32(0)) revert EmptyRoot();
        if (size <= r.lastSize) revert SizeNotIncreasing();
        if (prevRoot != r.lastRoot) revert PrevRootMismatch();
        if (rootAt_[runId][size] != bytes32(0)) revert AlreadyAnchored();

        rootAt_[runId][size] = root;
        r.lastSize = size;
        r.lastRoot = root;
        r.count += 1;
        emit Anchored(runId, size, root, prevRoot, msg.sender);
    }

    /// @notice Single-shot anchor (preregistration, run plan). The second call for an id reverts.
    function anchorOnce(bytes32 id, bytes32 digest) external {
        if (msg.sender != deployer) revert NotDeployer();
        if (digest == bytes32(0)) revert EmptyDigest();
        if (onceDigest_[id] != bytes32(0)) revert AlreadyAnchored();
        onceDigest_[id] = digest;
        emit AnchoredOnce(id, digest);
    }

    function runOwner(bytes32 runId) external view returns (address) {
        return runs[runId].owner;
    }

    function lastSize(bytes32 runId) external view returns (uint64) {
        return runs[runId].lastSize;
    }

    function lastRoot(bytes32 runId) external view returns (bytes32) {
        return runs[runId].lastRoot;
    }

    /// @notice Number of anchors written for the run.
    function anchorCount(bytes32 runId) external view returns (uint64) {
        return runs[runId].count;
    }

    function rootAt(bytes32 runId, uint64 size) external view returns (bytes32) {
        return rootAt_[runId][size];
    }

    function onceDigestOf(bytes32 id) external view returns (bytes32) {
        return onceDigest_[id];
    }

    /// @notice sha256(runId(32) || uint64_be(size) || root(32)); reverts if (runId,size) is not anchored.
    function digestOf(bytes32 runId, uint64 size) external view returns (bytes32) {
        bytes32 root = rootAt_[runId][size];
        if (root == bytes32(0)) revert NotAnchored();
        return sha256(abi.encodePacked(runId, size, root));
    }
}
