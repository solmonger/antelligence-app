const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ExperienceRegistry", function () {
  let registry, owner, validator, submitter;

  beforeEach(async function () {
    [owner, validator, submitter] = await ethers.getSigners();
    const ExperienceRegistry = await ethers.getContractFactory("ExperienceRegistry");
    registry = await ExperienceRegistry.deploy();
    await registry.addValidator(validator.address);
  });

  it("should set deployer as owner", async function () {
    expect(await registry.owner()).to.equal(owner.address);
  });

  it("should submit an experience", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("exp-1"));
    const dataHash = ethers.keccak256(ethers.toUtf8Bytes("data-1"));
    await registry.connect(submitter).submitExperience(
      runHash, "QmTest123", dataHash, 95,
      "pheromone-guided", "qwen3.5-9b", 10, 50, ethers.ZeroHash
    );
    const [exp] = await registry.getExperience(runHash);
    expect(exp.score).to.equal(95);
    expect(exp.submitter).to.equal(submitter.address);
  });

  it("should reject duplicate experience", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("exp-dup"));
    const dataHash = ethers.keccak256(ethers.toUtf8Bytes("data"));
    await registry.submitExperience(runHash, "Qm1", dataHash, 50, "a", "b", 1, 1, ethers.ZeroHash);
    await expect(
      registry.submitExperience(runHash, "Qm2", dataHash, 60, "a", "b", 1, 1, ethers.ZeroHash)
    ).to.be.revertedWith("Experience already exists");
  });

  it("should attest and auto-verify", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("exp-verify"));
    const dataHash = ethers.keccak256(ethers.toUtf8Bytes("data-v"));
    await registry.submitExperience(runHash, "Qm", dataHash, 80, "t", "m", 5, 20, ethers.ZeroHash);

    // Owner is also a validator by default
    await registry.connect(owner).attestExperience(runHash, 85, "good");
    await registry.connect(validator).attestExperience(runHash, 90, "excellent");

    expect(await registry.isVerified(runHash)).to.be.true;
    expect(await registry.calculateAverageQuality(runHash)).to.equal(87);
  });

  it("should reject non-validator attestation", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("exp-noauth"));
    const dataHash = ethers.keccak256(ethers.toUtf8Bytes("data-n"));
    await registry.submitExperience(runHash, "Qm", dataHash, 50, "t", "m", 1, 1, ethers.ZeroHash);
    await expect(
      registry.connect(submitter).attestExperience(runHash, 80, "")
    ).to.be.revertedWith("Not authorized validator");
  });
});
