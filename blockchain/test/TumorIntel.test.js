const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("TumorIntel", function () {
  let TumorIntel;
  let tumorIntel;
  let owner;
  let nanobot1;
  let nanobot2;
  let nanobot3;
  
  beforeEach(async function () {
    [owner, nanobot1, nanobot2, nanobot3] = await ethers.getSigners();
    
    TumorIntel = await ethers.getContractFactory("TumorIntel");
    tumorIntel = await TumorIntel.deploy();
    await tumorIntel.waitForDeployment();
  });
  
  describe("Deployment", function () {
    it("Should deploy with zero initial pins", async function () {
      expect(await tumorIntel.getPinCount()).to.equal(0);
    });
  });
  
  describe("Intel Reporting", function () {
    it("Should allow reporting new intel", async function () {
      await expect(
        tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 8) // HYPOXIC_CLUSTER
      ).to.emit(tumorIntel, "IntelReported")
       .withArgs(0, 100, 200, 0, nanobot1.address, 8);
      
      expect(await tumorIntel.getPinCount()).to.equal(1);
      
      const pin = await tumorIntel.intelPins(0);
      expect(pin.x).to.equal(100);
      expect(pin.y).to.equal(200);
      expect(pin.pinType).to.equal(0); // HYPOXIC_CLUSTER
      expect(pin.reporter).to.equal(nanobot1.address);
      expect(pin.priority).to.equal(8);
      expect(pin.isActive).to.equal(true);
    });
    
    it("Should reject invalid priority values", async function () {
      await expect(
        tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 0)
      ).to.be.revertedWith("Priority must be between 1 and 10");
      
      await expect(
        tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 11)
      ).to.be.revertedWith("Priority must be between 1 and 10");
    });
    
    it("Should report multiple pins", async function () {
      await tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 8);
      await tumorIntel.connect(nanobot2).reportIntel(150, 250, 1, 9); // STEM_CELL_DETECTED
      await tumorIntel.connect(nanobot3).reportIntel(200, 300, 2, 7); // HIGH_RESISTANCE_AREA
      
      expect(await tumorIntel.getPinCount()).to.equal(3);
    });
  });
  
  describe("Intel Confirmation", function () {
    beforeEach(async function () {
      await tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 8);
    });
    
    it("Should allow confirming intel", async function () {
      await expect(
        tumorIntel.connect(nanobot2).confirmIntel(0)
      ).to.emit(tumorIntel, "IntelConfirmed")
       .withArgs(0, nanobot2.address, 1);
      
      expect(await tumorIntel.confirmations(0)).to.equal(1);
      expect(await tumorIntel.hasConfirmed(0, nanobot2.address)).to.equal(true);
    });
    
    it("Should reject confirming invalid pin", async function () {
      await expect(
        tumorIntel.connect(nanobot2).confirmIntel(999)
      ).to.be.revertedWith("Invalid pin ID");
    });
    
    it("Should reject confirming deactivated intel", async function () {
      await tumorIntel.connect(nanobot1).deactivateIntel(0);
      
      await expect(
        tumorIntel.connect(nanobot2).confirmIntel(0)
      ).to.be.revertedWith("Intel is no longer active");
    });
    
    it("Should reject double confirmation", async function () {
      await tumorIntel.connect(nanobot2).confirmIntel(0);
      
      await expect(
        tumorIntel.connect(nanobot2).confirmIntel(0)
      ).to.be.revertedWith("Already confirmed this intel");
    });
    
    it("Should track multiple confirmations", async function () {
      await tumorIntel.connect(nanobot2).confirmIntel(0);
      await tumorIntel.connect(nanobot3).confirmIntel(0);
      
      expect(await tumorIntel.confirmations(0)).to.equal(2);
      expect(await tumorIntel.hasConfirmed(0, nanobot2.address)).to.equal(true);
      expect(await tumorIntel.hasConfirmed(0, nanobot3.address)).to.equal(true);
    });
  });
  
  describe("Intel Deactivation", function () {
    beforeEach(async function () {
      await tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 8);
      await tumorIntel.connect(nanobot2).confirmIntel(0);
    });
    
    it("Should allow reporter to deactivate", async function () {
      await expect(
        tumorIntel.connect(nanobot1).deactivateIntel(0)
      ).to.emit(tumorIntel, "IntelDeactivated")
       .withArgs(0, nanobot1.address);
      
      const pin = await tumorIntel.intelPins(0);
      expect(pin.isActive).to.equal(false);
    });
    
    it("Should allow confirmer to deactivate", async function () {
      await expect(
        tumorIntel.connect(nanobot2).deactivateIntel(0)
      ).to.emit(tumorIntel, "IntelDeactivated")
       .withArgs(0, nanobot2.address);
    });
    
    it("Should reject unauthorized deactivation", async function () {
      await expect(
        tumorIntel.connect(nanobot3).deactivateIntel(0)
      ).to.be.revertedWith("Not authorized to deactivate");
    });
    
    it("Should reject deactivating already deactivated intel", async function () {
      await tumorIntel.connect(nanobot1).deactivateIntel(0);
      
      await expect(
        tumorIntel.connect(nanobot1).deactivateIntel(0)
      ).to.be.revertedWith("Intel already deactivated");
    });
  });
  
  describe("Priority Updates", function () {
    beforeEach(async function () {
      await tumorIntel.connect(nanobot1).reportIntel(100, 200, 0, 8);
      await tumorIntel.connect(nanobot2).confirmIntel(0);
    });
    
    it("Should allow reporter to update priority", async function () {
      await expect(
        tumorIntel.connect(nanobot1).updateIntelPriority(0, 10)
      ).to.emit(tumorIntel, "IntelPriorityUpdated")
       .withArgs(0, 8, 10, nanobot1.address);
      
      const pin = await tumorIntel.intelPins(0);
      expect(pin.priority).to.equal(10);
    });
    
    it("Should allow confirmer to update priority", async function () {
      await expect(
        tumorIntel.connect(nanobot2).updateIntelPriority(0, 5)
      ).to.emit(tumorIntel, "IntelPriorityUpdated")
       .withArgs(0, 8, 5, nanobot2.address);
    });
    
    it("Should reject unauthorized priority updates", async function () {
      await expect(
        tumorIntel.connect(nanobot3).updateIntelPriority(0, 10)
      ).to.be.revertedWith("Not authorized to update priority");
    });
    
    it("Should reject invalid priority values", async function () {
      await expect(
        tumorIntel.connect(nanobot1).updateIntelPriority(0, 0)
      ).to.be.revertedWith("Priority must be between 1 and 10");
      
      await expect(
        tumorIntel.connect(nanobot1).updateIntelPriority(0, 11)
      ).to.be.revertedWith("Priority must be between 1 and 10");
    });
    
    it("Should reject updating deactivated intel", async function () {
      await tumorIntel.connect(nanobot1).deactivateIntel(0);
      
      await expect(
        tumorIntel.connect(nanobot1).updateIntelPriority(0, 10)
      ).to.be.revertedWith("Intel is no longer active");
    });
  });
  
  describe("Query Functions", function () {
    beforeEach(async function () {
      // Create various pins
      await tumorIntel.connect(nanobot1).reportIntel(100, 100, 0, 8); // HYPOXIC_CLUSTER
      await tumorIntel.connect(nanobot2).reportIntel(200, 200, 1, 9); // STEM_CELL_DETECTED
      await tumorIntel.connect(nanobot3).reportIntel(300, 300, 0, 7); // HYPOXIC_CLUSTER
      await tumorIntel.connect(nanobot1).reportIntel(400, 400, 2, 6); // HIGH_RESISTANCE_AREA
      
      // Deactivate one pin
      await tumorIntel.connect(nanobot3).deactivateIntel(2);
    });
    
    it("Should get active pins", async function () {
      const activePins = await tumorIntel.getActivePins();
      expect(activePins.length).to.equal(3); // One was deactivated
      expect(activePins[0]).to.equal(0);
      expect(activePins[1]).to.equal(1);
      expect(activePins[2]).to.equal(3);
    });
    
    it("Should get pins by type", async function () {
      const hypoxicPins = await tumorIntel.getPinsByType(0); // HYPOXIC_CLUSTER
      expect(hypoxicPins.length).to.equal(1); // Only pin 0 is active and HYPOXIC_CLUSTER
      expect(hypoxicPins[0]).to.equal(0);
    });
    
    it("Should get high priority pins", async function () {
      const highPriorityPins = await tumorIntel.getHighPriorityPins();
      expect(highPriorityPins.length).to.equal(2); // Pins 0 and 1 have priority >= 8
      expect(highPriorityPins[0]).to.equal(0);
      expect(highPriorityPins[1]).to.equal(1);
    });
    
    it("Should get pins in area", async function () {
      // Area around (150, 150) with radius 100 should include pins 0 and 1
      const pinsInArea = await tumorIntel.getPinsInArea(150, 150, 100);
      expect(pinsInArea.length).to.equal(2);
      
      // Verify the pins are 0 and 1 (order not guaranteed)
      const pinIds = pinsInArea.map(id => Number(id));
      expect(pinIds).to.include(0);
      expect(pinIds).to.include(1);
    });
    
    it("Should get empty array for area with no pins", async function () {
      const pinsInArea = await tumorIntel.getPinsInArea(500, 500, 50);
      expect(pinsInArea.length).to.equal(0);
    });
  });
});