const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("FoodToken", function () {
  let food, colony, ant1;

  beforeEach(async function () {
    [colony, ant1] = await ethers.getSigners();
    const FoodToken = await ethers.getContractFactory("FoodToken");
    food = await FoodToken.deploy(colony.address);
  });

  it("should set colony address", async function () {
    expect(await food.colony()).to.equal(colony.address);
  });

  it("should mint token to ant", async function () {
    await food.connect(colony).mint(ant1.address);
    expect(await food.ownerOf(0)).to.equal(ant1.address);
    expect(await food.nextId()).to.equal(1);
  });

  it("should reject mint from non-colony", async function () {
    await expect(food.connect(ant1).mint(ant1.address)).to.be.revertedWith("Only colony may mint");
  });

  it("should mint sequential token IDs", async function () {
    await food.connect(colony).mint(ant1.address);
    await food.connect(colony).mint(ant1.address);
    expect(await food.ownerOf(0)).to.equal(ant1.address);
    expect(await food.ownerOf(1)).to.equal(ant1.address);
    expect(await food.nextId()).to.equal(2);
  });

  it("should have correct name and symbol", async function () {
    expect(await food.name()).to.equal("AntelligenceFood");
    expect(await food.symbol()).to.equal("FOOD");
  });
});
