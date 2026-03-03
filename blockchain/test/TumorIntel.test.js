const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TumorIntel", function () {
  let intel, owner, nanobot1, nanobot2;

  beforeEach(async function () {
    [owner, nanobot1, nanobot2] = await ethers.getSigners();
    const TumorIntel = await ethers.getContractFactory("TumorIntel");
    intel = await TumorIntel.deploy();
  });

  it("should report intel and return pin ID", async function () {
    const tx = await intel.connect(nanobot1).reportIntel(100, 200, 0, 5);
    const receipt = await tx.wait();
    expect(await intel.getIntelCount()).to.equal(1);
  });

  it("should reject invalid priority", async function () {
    await expect(intel.reportIntel(0, 0, 0, 0)).to.be.revertedWith("Priority must be between 1 and 10");
    await expect(intel.reportIntel(0, 0, 0, 11)).to.be.revertedWith("Priority must be between 1 and 10");
  });

  it("should confirm intel and track confirmations", async function () {
    await intel.connect(nanobot1).reportIntel(50, 50, 1, 8);
    await intel.connect(nanobot2).confirmIntel(0);
    expect(await intel.confirmations(0)).to.equal(1);
  });

  it("should prevent double confirmation", async function () {
    await intel.connect(nanobot1).reportIntel(10, 10, 2, 3);
    await intel.connect(nanobot2).confirmIntel(0);
    await expect(intel.connect(nanobot2).confirmIntel(0)).to.be.revertedWith("Already confirmed this intel");
  });

  it("should deactivate intel", async function () {
    await intel.connect(nanobot1).reportIntel(1, 1, 4, 7);
    await intel.deactivateIntel(0);
    const details = await intel.getIntelDetails(0);
    expect(details.isActive).to.be.false;
  });

  it("should return active intel by type", async function () {
    await intel.reportIntel(1, 1, 0, 5); // HYPOXIC_CLUSTER
    await intel.reportIntel(2, 2, 1, 5); // STEM_CELL_DETECTED
    await intel.reportIntel(3, 3, 0, 5); // HYPOXIC_CLUSTER
    const active = await intel.getActiveIntelByType(0);
    expect(active.length).to.equal(2);
  });
});
