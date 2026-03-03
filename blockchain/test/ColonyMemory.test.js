const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("ColonyMemory", function () {
  let colony, owner, ant1;

  beforeEach(async function () {
    [owner, ant1] = await ethers.getSigners();
    const ColonyMemory = await ethers.getContractFactory("ColonyMemory");
    colony = await ColonyMemory.deploy();
  });

  it("should mark a cell as visited", async function () {
    await colony.connect(ant1).markVisited(10, 20);
    expect(await colony.hasVisited(10, 20)).to.be.true;
  });

  it("should not re-emit event for already visited cell", async function () {
    await colony.connect(ant1).markVisited(5, 5);
    // Second visit to same cell should not emit
    await expect(colony.connect(ant1).markVisited(5, 5)).to.not.emit(colony, "CellVisited");
  });

  it("should emit FoodCollected on recordFood", async function () {
    await expect(colony.connect(ant1).recordFood(1, 3, 4))
      .to.emit(colony, "FoodCollected")
      .withArgs(1, 3, 4, ant1.address);
  });

  it("should initialize and complete a simulation", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("sim-1"));
    await colony.initializeSimulation(runHash);
    const sim = await colony.getSimulationRun(runHash);
    expect(sim.submitter).to.equal(owner.address);
    expect(sim.completed).to.be.false;

    await colony.completeSimulation(runHash, 100, 5, 10);
    const completed = await colony.getSimulationRun(runHash);
    expect(completed.completed).to.be.true;
    expect(completed.cellsKilled).to.equal(5);
  });

  it("should record drug delivery and tumor kill", async function () {
    const runHash = ethers.keccak256(ethers.toUtf8Bytes("sim-2"));
    await colony.initializeSimulation(runHash);

    await colony.recordDrugDelivery(runHash, 1, 2, 3, 100, 50);
    expect(await colony.getDeliveryCount(runHash)).to.equal(1);

    await colony.recordTumorKill(runHash, 42, 1, 2, 3, 200);
    expect(await colony.getKillCount(runHash)).to.equal(1);
  });
});
