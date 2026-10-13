// v4.1 contracts: AnchorRegistry, BoardLog, ReputationRegistry (docs/v4.1/INTERFACES.md sections 3, 4).
const { expect } = require("chai");
const { ethers } = require("hardhat");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const ZERO = ethers.ZeroHash;
const sha = (b) => "0x" + crypto.createHash("sha256").update(b).digest("hex");
const h = (label) => sha(Buffer.from(label));
const coder = ethers.AbiCoder.defaultAbiCoder();

// Anchor-digest vectors come from tests/v4_1/vectors/interfaces.json (owned by L3), key `anchor.digests`:
// [{run_id, size, root, preimage_hex, digest}] (hex, no 0x). The file is required: no fallback.
function interfaceAnchorCases() {
  const file = path.join(__dirname, "..", "..", "tests", "v4_1", "vectors", "interfaces.json");
  return JSON.parse(fs.readFileSync(file, "utf8")).anchor.digests;
}

function digestVector(runId, size, root) {
  const sz = Buffer.alloc(8);
  sz.writeBigUInt64BE(BigInt(size));
  return sha(Buffer.concat([ethers.getBytes(runId), sz, ethers.getBytes(root)]));
}

const TYPE = { task: 0, knowledge: 1, evidence: 2, route: 3, decision: 4, edit: 5, alarm: 6 };
function compact(seq, typeCode, id, prevHash, payloadInts = [1, 2]) {
  const payload = coder.encode(payloadInts.map(() => "uint256"), payloadInts);
  return coder.encode(["uint64", "uint8", "bytes32", "bytes32", "bytes"], [seq, typeCode, id, prevHash, payload]);
}
function chainEntries(n, firstSeq = 0, prev = ZERO, tag = "e") {
  const out = [];
  const ids = [];
  let last = prev;
  for (let i = 0; i < n; i++) {
    const id = h(`${tag}-${firstSeq + i}`);
    out.push(compact(firstSeq + i, i % 7, id, last));
    ids.push(id);
    last = id;
  }
  return { entries: out, ids, last };
}

describe("AnchorRegistry", function () {
  let reg, owner, other;
  const run = h("run-1");
  beforeEach(async function () {
    [owner, other] = await ethers.getSigners();
    reg = await (await ethers.getContractFactory("AnchorRegistry")).deploy();
  });

  it("anchors monotonic sizes with a prevRoot chain and emits Anchored", async function () {
    const r50 = h("root-50"), r100 = h("root-100");
    await expect(reg.anchor(run, 50, r50, ZERO))
      .to.emit(reg, "Anchored").withArgs(run, 50, r50, ZERO, owner.address);
    await reg.anchor(run, 100, r100, r50);
    expect(await reg.anchorCount(run)).to.equal(2);
    expect(await reg.lastSize(run)).to.equal(100);
    expect(await reg.lastRoot(run)).to.equal(r100);
    expect(await reg.rootAt(run, 50)).to.equal(r50);
    expect(await reg.runOwner(run)).to.equal(owner.address);
  });

  it("rejects non-increasing size, wrong prevRoot, empty root", async function () {
    const r50 = h("root-50");
    await reg.anchor(run, 50, r50, ZERO);
    await expect(reg.anchor(run, 50, h("x"), r50)).to.be.revertedWithCustomError(reg, "SizeNotIncreasing");
    await expect(reg.anchor(run, 40, h("x"), r50)).to.be.revertedWithCustomError(reg, "SizeNotIncreasing");
    await expect(reg.anchor(run, 100, h("x"), ZERO)).to.be.revertedWithCustomError(reg, "PrevRootMismatch");
    await expect(reg.anchor(run, 100, h("x"), h("wrong"))).to.be.revertedWithCustomError(reg, "PrevRootMismatch");
    await expect(reg.anchor(run, 100, ZERO, r50)).to.be.revertedWithCustomError(reg, "EmptyRoot");
  });

  it("first prevRoot must be zero", async function () {
    await expect(reg.anchor(run, 50, h("r"), h("not-zero"))).to.be.revertedWithCustomError(reg, "PrevRootMismatch");
  });

  it("write-once per (runId,size): a stored root can never change", async function () {
    const r50 = h("root-50");
    await reg.anchor(run, 50, r50, ZERO);
    await reg.anchor(run, 100, h("root-100"), r50);
    // Re-anchoring an older size is blocked by monotonicity; the stored root is untouched.
    await expect(reg.anchor(run, 50, h("other"), h("root-100"))).to.be.revertedWithCustomError(reg, "SizeNotIncreasing");
    expect(await reg.rootAt(run, 50)).to.equal(r50);
  });

  it("anchor is deployer-only: a third party cannot squat a runId", async function () {
    const squat = h("run-squat");
    await expect(reg.connect(other).anchor(squat, 50, h("evil"), ZERO)).to.be.revertedWithCustomError(reg, "NotDeployer");
    expect(await reg.runOwner(squat)).to.equal(ethers.ZeroAddress);
    // The real anchor for that runId still works afterwards.
    await reg.anchor(squat, 50, h("real"), ZERO);
    expect(await reg.runOwner(squat)).to.equal(owner.address);
    await expect(reg.connect(other).anchor(squat, 100, h("r100"), h("real"))).to.be.revertedWithCustomError(reg, "NotDeployer");
  });

  it("anchorOnce succeeds once and reverts the second time", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("v4.1|prereg"));
    const digest = h("prereg-doc");
    await expect(reg.anchorOnce(id, digest)).to.emit(reg, "AnchoredOnce").withArgs(id, digest);
    expect(await reg.onceDigestOf(id)).to.equal(digest);
    await expect(reg.anchorOnce(id, digest)).to.be.revertedWithCustomError(reg, "AlreadyAnchored");
    await expect(reg.anchorOnce(id, h("different"))).to.be.revertedWithCustomError(reg, "AlreadyAnchored");
    const id2 = ethers.keccak256(ethers.toUtf8Bytes("v4.1|runplan"));
    await reg.anchorOnce(id2, h("plan"));
  });

  it("anchorOnce is deployer-only and rejects an empty digest", async function () {
    const id = ethers.keccak256(ethers.toUtf8Bytes("v4.1|prereg"));
    await expect(reg.connect(other).anchorOnce(id, h("d"))).to.be.revertedWithCustomError(reg, "NotDeployer");
    await expect(reg.anchorOnce(id, ZERO)).to.be.revertedWithCustomError(reg, "EmptyDigest");
  });

  it("digestOf equals sha256(runId || uint64_be(size) || root)", async function () {
    const root = h("root-50");
    await reg.anchor(run, 50, root, ZERO);
    expect(await reg.digestOf(run, 50)).to.equal(digestVector(run, 50, root));
    // A size that does not fit in one byte and a large size, to pin the big-endian width.
    await reg.anchor(run, 0x0102030405n, h("big"), root);
    expect(await reg.digestOf(run, 0x0102030405n)).to.equal(digestVector(run, 0x0102030405n, h("big")));
    await expect(reg.digestOf(run, 51)).to.be.revertedWithCustomError(reg, "NotAnchored");
  });

  it("digestOf matches the L3 interfaces.json anchor.digests vectors", async function () {
    const cases = interfaceAnchorCases();
    expect(cases.length).to.be.greaterThan(0);
    const last = {};
    for (const c of cases) {
      const runId = "0x" + c.run_id, root = "0x" + c.root;
      await reg.anchor(runId, c.size, root, last[runId] || ZERO); // vectors are increasing sizes per run
      last[runId] = root;
      expect(await reg.digestOf(runId, c.size)).to.equal("0x" + c.digest);
      expect(digestVector(runId, c.size, root)).to.equal("0x" + c.digest);
    }
  });
});

describe("BoardLog", function () {
  let log, owner, other;
  const run = h("run-1");
  beforeEach(async function () {
    [owner, other] = await ethers.getSigners();
    log = await (await ethers.getContractFactory("BoardLog")).deploy();
  });

  it("posts a contiguous batch, emits one Entry per entry, advances state", async function () {
    const { entries, ids, last } = chainEntries(5);
    const tx = await log.postBatch(run, 0, ZERO, entries);
    const rc = await tx.wait();
    const evs = rc.logs.map((l) => log.interface.parseLog(l)).filter((e) => e && e.name === "Entry");
    expect(evs.length).to.equal(5);
    evs.forEach((e, i) => {
      expect(e.args.runId).to.equal(run);
      expect(e.args.seq).to.equal(i);
      expect(e.args.id).to.equal(ids[i]);
      expect(e.args.typeCode).to.equal(i % 7);
      expect(e.args.data).to.equal(coder.encode(["uint256", "uint256"], [1, 2]));
    });
    expect(await log.nextSeq(run)).to.equal(5);
    expect(await log.lastId(run)).to.equal(last);
    expect(await log.runOwner(run)).to.equal(owner.address);
  });

  it("continues across batches and rejects gaps / replays", async function () {
    const a = chainEntries(3);
    await log.postBatch(run, 0, ZERO, a.entries);
    const b = chainEntries(2, 3, a.last);
    await expect(log.postBatch(run, 4, a.last, b.entries)).to.be.revertedWithCustomError(log, "SeqMismatch");
    await expect(log.postBatch(run, 0, ZERO, a.entries)).to.be.revertedWithCustomError(log, "SeqMismatch");
    await log.postBatch(run, 3, a.last, b.entries);
    expect(await log.nextSeq(run)).to.equal(5);
  });

  it("a duplicate batch reverts and emits nothing (replay-safe)", async function () {
    const { entries, last } = chainEntries(3);
    await log.postBatch(run, 0, ZERO, entries);
    await expect(log.postBatch(run, 0, ZERO, entries)).to.be.revertedWithCustomError(log, "SeqMismatch");
    // Also a partial overlap: restart from seq 2 with the right prevHash of a stale view.
    await expect(log.postBatch(run, 2, ZERO, entries.slice(2))).to.be.revertedWithCustomError(log, "SeqMismatch");
    expect(await log.nextSeq(run)).to.equal(3);
    expect(await log.lastId(run)).to.equal(last);
  });

  it("rejects a wrong batch prevHash", async function () {
    const a = chainEntries(2);
    await log.postBatch(run, 0, ZERO, a.entries);
    const b = chainEntries(2, 2, a.last);
    await expect(log.postBatch(run, 2, h("wrong"), b.entries)).to.be.revertedWithCustomError(log, "PrevHashMismatch");
    await expect(log.postBatch(run, 2, ZERO, b.entries)).to.be.revertedWithCustomError(log, "PrevHashMismatch");
  });

  it("rejects an entry whose prevHash does not link to the previous id (inside a batch)", async function () {
    const ids = [h("a"), h("b"), h("c")];
    const entries = [
      compact(0, 0, ids[0], ZERO),
      compact(1, 1, ids[1], ids[0]),
      compact(2, 2, ids[2], h("not-b")),
    ];
    await expect(log.postBatch(run, 0, ZERO, entries)).to.be.revertedWithCustomError(log, "PrevHashMismatch").withArgs(2);
    // Nothing was committed.
    expect(await log.nextSeq(run)).to.equal(0);
  });

  it("rejects a seq gap inside a batch, a bad type code, an empty id and an empty batch", async function () {
    await expect(log.postBatch(run, 0, ZERO, [compact(0, 0, h("a"), ZERO), compact(2, 0, h("b"), h("a"))]))
      .to.be.revertedWithCustomError(log, "SeqMismatch");
    await expect(log.postBatch(run, 0, ZERO, [compact(0, 7, h("a"), ZERO)])).to.be.revertedWithCustomError(log, "BadTypeCode");
    await expect(log.postBatch(run, 0, ZERO, [compact(0, 0, ZERO, ZERO)])).to.be.revertedWithCustomError(log, "EmptyId");
    await expect(log.postBatch(run, 0, ZERO, [])).to.be.revertedWithCustomError(log, "EmptyBatch");
  });

  it("postBatch is deployer-only: a third party can neither squat a runId nor extend one", async function () {
    expect(await log.deployer()).to.equal(owner.address);
    const a = chainEntries(1);
    // squat attempt: first post by a stranger reverts and leaves the run unowned
    await expect(log.connect(other).postBatch(run, 0, ZERO, a.entries)).to.be.revertedWithCustomError(log, "NotDeployer");
    expect(await log.runOwner(run)).to.equal(ethers.ZeroAddress);
    expect(await log.nextSeq(run)).to.equal(0);
    // the deployer is not locked out, and the owner is always the deployer
    await log.postBatch(run, 0, ZERO, a.entries);
    expect(await log.runOwner(run)).to.equal(owner.address);
    // a stranger cannot extend the run either
    const b = chainEntries(1, 1, a.last);
    await expect(log.connect(other).postBatch(run, 1, a.last, b.entries)).to.be.revertedWithCustomError(log, "NotDeployer");
    expect(await log.nextSeq(run)).to.equal(1);
  });

  it("covers every type code", async function () {
    const { entries } = chainEntries(7);
    const rc = await (await log.postBatch(run, 0, ZERO, entries)).wait();
    const codes = rc.logs.map((l) => Number(log.interface.parseLog(l).args.typeCode));
    expect(codes).to.deep.equal(Object.values(TYPE));
  });
});

describe("ReputationRegistry", function () {
  let rep, owner, other;
  const run = h("run-1");
  const author = h("squad-3");
  let claimN = 0;
  const claim = () => h(`claim-${claimN++}`);
  const table = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "..", "tests", "v4_1", "vectors", "reputation_parity.json"), "utf8"),
  );
  beforeEach(async function () {
    [owner, other] = await ethers.getSigners();
    rep = await (await ethers.getContractFactory("ReputationRegistry")).deploy();
    claimN = 0;
  });

  it("parity table (shared with the Python rule): update(r, contradicted) == expect", async function () {
    expect(table.cases.length).to.be.greaterThan(10);
    for (const c of table.cases) {
      expect(await rep.update(c.r, c.contradicted), `r=${c.r} contradicted=${c.contradicted}`).to.equal(c.expect);
    }
  });

  it("constants and exclusion threshold match the table", async function () {
    expect(await rep.START_PPM()).to.equal(table.start_ppm);
    expect(await rep.EXCLUDE_BELOW_PPM()).to.equal(table.exclude_below_ppm);
  });

  it("starts at 1e6, halves on contradiction, recovers by 100k, caps at 1e6", async function () {
    expect(await rep.reputationOf(author)).to.equal(1_000_000);
    await expect(rep.report(run, author, claim(), true)).to.emit(rep, "Reported").withArgs(run, author, h("claim-0"), true, 500_000);
    await rep.report(run, author, claim(), true); // 250000 -> excluded
    expect(await rep.reputationOf(author)).to.equal(250_000);
    expect(await rep.isExcluded(author)).to.equal(true);
    await rep.report(run, author, claim(), false); // 350000
    expect(await rep.isExcluded(author)).to.equal(false);
    for (let i = 0; i < 10; i++) await rep.report(run, author, claim(), false);
    expect(await rep.reputationOf(author)).to.equal(1_000_000);
  });

  it("emits Excluded exactly when the score crosses below the threshold", async function () {
    await rep.report(run, author, claim(), true); // 500000
    await expect(rep.report(run, author, claim(), true)).to.emit(rep, "Excluded").withArgs(author, 250_000);
    await expect(rep.report(run, author, claim(), true)).to.not.emit(rep, "Excluded"); // already excluded
  });

  it("walks a sequence the same as the table rule (running state)", async function () {
    const seq = [true, false, true, true, false, false, true];
    let r = 1_000_000;
    for (const c of seq) {
      r = Number(await rep.update(r, c));
      await rep.report(run, author, claim(), c);
      expect(await rep.reputationOf(author)).to.equal(r);
    }
  });

  it("report is idempotent: a duplicate (runId, author, claimId) reverts and changes nothing", async function () {
    const c = h("claim-dup");
    expect(await rep.reported(run, author, c)).to.equal(false);
    await rep.report(run, author, c, true);
    expect(await rep.reported(run, author, c)).to.equal(true);
    expect(await rep.reputationOf(author)).to.equal(500_000);
    await expect(rep.report(run, author, c, true)).to.be.revertedWithCustomError(rep, "DuplicateReport");
    await expect(rep.report(run, author, c, false)).to.be.revertedWithCustomError(rep, "DuplicateReport");
    expect(await rep.reputationOf(author)).to.equal(500_000);
    // Same claim id under another run or author is a different claim.
    await rep.report(h("run-2"), author, c, false);
    await rep.report(run, h("squad-4"), c, false);
  });

  it("only the reporter can report", async function () {
    await expect(rep.connect(other).report(run, author, h("c"), true)).to.be.revertedWithCustomError(rep, "NotReporter");
  });
});
