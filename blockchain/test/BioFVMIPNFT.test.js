const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("BioFVMIPNFT", function () {
  let ipnft, owner, addr1, addr2;
  const sampleCID = "bafkreigan5c6ndz3mncjnddloyqtzbwz72h4r2hqpa7bzef5fjqzfbydty";
  const sampleHash = ethers.keccak256(ethers.toUtf8Bytes("sample-config"));
  const sampleSchemaUID = ethers.keccak256(ethers.toUtf8Bytes("biofvm-schema"));
  const sampleAttestationUID = ethers.keccak256(ethers.toUtf8Bytes("attestation-1"));

  beforeEach(async function () {
    [owner, addr1, addr2] = await ethers.getSigners();
    const BioFVMIPNFT = await ethers.getContractFactory("BioFVMIPNFT");
    ipnft = await BioFVMIPNFT.deploy();
    await ipnft.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set name and symbol", async function () {
      expect(await ipnft.name()).to.equal("BioFVM Simulation IP");
      expect(await ipnft.symbol()).to.equal("BFVM-IP");
    });

    it("Should set owner", async function () {
      expect(await ipnft.owner()).to.equal(owner.address);
    });

    it("Should start with zero supply", async function () {
      expect(await ipnft.totalSupply()).to.equal(0);
    });
  });

  describe("Minting", function () {
    it("Should mint an IP-NFT", async function () {
      const tx = await ipnft.mint(
        addr1.address,
        "ipfs://metadata-uri",
        sampleCID,
        sampleHash,
        "Glioblastoma tumor growth model",
        sampleSchemaUID
      );

      await expect(tx)
        .to.emit(ipnft, "ConfigMinted")
        .withArgs(0, addr1.address, sampleCID, sampleHash, "Glioblastoma tumor growth model");

      expect(await ipnft.ownerOf(0)).to.equal(addr1.address);
      expect(await ipnft.totalSupply()).to.equal(1);
    });

    it("Should store config metadata", async function () {
      await ipnft.mint(
        addr1.address, "ipfs://meta", sampleCID, sampleHash,
        "Test config", sampleSchemaUID
      );

      const config = await ipnft.configs(0);
      expect(config.ipfsCid).to.equal(sampleCID);
      expect(config.configHash).to.equal(sampleHash);
      expect(config.description).to.equal("Test config");
      expect(config.easSchemaUID).to.equal(sampleSchemaUID);
      expect(config.mintedAt).to.be.gt(0);
    });

    it("Should increment token IDs", async function () {
      await ipnft.mint(addr1.address, "ipfs://1", sampleCID, sampleHash, "Config 1", sampleSchemaUID);
      await ipnft.mint(addr2.address, "ipfs://2", sampleCID, sampleHash, "Config 2", sampleSchemaUID);

      expect(await ipnft.ownerOf(0)).to.equal(addr1.address);
      expect(await ipnft.ownerOf(1)).to.equal(addr2.address);
      expect(await ipnft.totalSupply()).to.equal(2);
    });

    it("Should set token URI", async function () {
      await ipnft.mint(addr1.address, "ipfs://test-metadata", sampleCID, sampleHash, "Test", sampleSchemaUID);
      expect(await ipnft.tokenURI(0)).to.equal("ipfs://test-metadata");
    });
  });

  describe("Attestation Linking", function () {
    beforeEach(async function () {
      await ipnft.mint(addr1.address, "ipfs://meta", sampleCID, sampleHash, "Test", sampleSchemaUID);
    });

    it("Should link an attestation", async function () {
      const tx = await ipnft.linkAttestation(0, sampleAttestationUID);
      await expect(tx)
        .to.emit(ipnft, "AttestationLinked")
        .withArgs(0, sampleAttestationUID);
    });

    it("Should store multiple attestations", async function () {
      const att2 = ethers.keccak256(ethers.toUtf8Bytes("attestation-2"));
      await ipnft.linkAttestation(0, sampleAttestationUID);
      await ipnft.linkAttestation(0, att2);

      const atts = await ipnft.getAttestations(0);
      expect(atts.length).to.equal(2);
      expect(atts[0]).to.equal(sampleAttestationUID);
      expect(atts[1]).to.equal(att2);
    });

    it("Should allow anyone to link attestations", async function () {
      await ipnft.connect(addr2).linkAttestation(0, sampleAttestationUID);
      const atts = await ipnft.getAttestations(0);
      expect(atts.length).to.equal(1);
    });

    it("Should reject linking to non-existent token", async function () {
      await expect(
        ipnft.linkAttestation(999, sampleAttestationUID)
      ).to.be.revertedWith("Token does not exist");
    });
  });
});
