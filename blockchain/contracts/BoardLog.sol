// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title BoardLog (Antelligence v4.1, mode D)
/// @notice Posts the compact encoding of board entries (docs/v4.1/INTERFACES.md section 4):
///         abi.encode(uint64 seq, uint8 typeCode, bytes32 id, bytes32 prevHash, bytes payloadCompact).
///         The contract checks that seq is contiguous and that every entry's prevHash is the id of
///         the entry before it. It cannot recompute ids (they are sha256 over JCS JSON), so the id
///         is a commitment that a verifier checks off-chain against the exported log.
contract BoardLog {
    uint8 public constant MAX_TYPE_CODE = 6; // task 0 ... alarm 6

    struct Run {
        address owner;
        uint64 nextSeq;
        bytes32 lastId;
    }

    mapping(bytes32 => Run) private runs;

    event Entry(bytes32 indexed runId, uint64 seq, bytes32 id, uint8 typeCode, bytes data);

    error NotRunOwner();
    error EmptyBatch();
    error SeqMismatch(uint64 expected, uint64 got);
    error PrevHashMismatch(uint64 seq);
    error BadTypeCode(uint64 seq);
    error EmptyId(uint64 seq);

    function postBatch(bytes32 runId, uint64 firstSeq, bytes32 prevHash, bytes[] calldata entries) external {
        Run storage r = runs[runId];
        if (r.owner == address(0)) {
            r.owner = msg.sender; // first post fixes the run owner
        } else if (r.owner != msg.sender) {
            revert NotRunOwner();
        }
        if (entries.length == 0) revert EmptyBatch();
        if (firstSeq != r.nextSeq) revert SeqMismatch(r.nextSeq, firstSeq);
        if (prevHash != r.lastId) revert PrevHashMismatch(firstSeq);

        bytes32 last = r.lastId;
        uint64 seq = firstSeq;
        for (uint256 i = 0; i < entries.length; i++) {
            (uint64 eSeq, uint8 typeCode, bytes32 id, bytes32 ePrev, bytes memory data) =
                abi.decode(entries[i], (uint64, uint8, bytes32, bytes32, bytes));
            if (eSeq != seq) revert SeqMismatch(seq, eSeq);
            if (ePrev != last) revert PrevHashMismatch(seq);
            if (typeCode > MAX_TYPE_CODE) revert BadTypeCode(seq);
            if (id == bytes32(0)) revert EmptyId(seq);
            emit Entry(runId, seq, id, typeCode, data);
            last = id;
            seq += 1;
        }
        r.nextSeq = seq;
        r.lastId = last;
    }

    function runOwner(bytes32 runId) external view returns (address) {
        return runs[runId].owner;
    }

    function nextSeq(bytes32 runId) external view returns (uint64) {
        return runs[runId].nextSeq;
    }

    function lastId(bytes32 runId) external view returns (bytes32) {
        return runs[runId].lastId;
    }
}
