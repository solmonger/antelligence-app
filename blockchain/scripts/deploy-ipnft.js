const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying BioFVMIPNFT with:", deployer.address);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Balance:", ethers.formatEther(balance), "ETH");

  const BioFVMIPNFT = await ethers.getContractFactory("BioFVMIPNFT");
  const ipnft = await BioFVMIPNFT.deploy();
  await ipnft.waitForDeployment();

  const address = await ipnft.getAddress();
  console.log("BioFVMIPNFT deployed:", address);
  console.log("View: https://sepolia.basescan.org/address/" + address);

  // Mint first IP-NFT as proof of concept
  console.log("\nMinting first IP-NFT...");
  const tx = await ipnft.mint(
    deployer.address,
    "ipfs://bafkreigan5c6ndz3mncjnddloyqtzbwz72h4r2hqpa7bzef5fjqzfbydty",
    "bafkreigan5c6ndz3mncjnddloyqtzbwz72h4r2hqpa7bzef5fjqzfbydty",
    ethers.keccak256(ethers.toUtf8Bytes("antelligence-default-tumor-config")),
    "Default glioblastoma nanobot simulation (10 bots, 150um radius)",
    "0xeb32221490772f15acecbb4ab0c154877e4953ad56cf5f0932f5d9f48f9e8a33" // EAS schema
  );
  await tx.wait();

  console.log("First IP-NFT minted! Token ID: 0");
  console.log("Total supply:", (await ipnft.totalSupply()).toString());

  // Link the existing attestation
  console.log("\nLinking EAS attestation...");
  const linkTx = await ipnft.linkAttestation(
    0,
    "0x4657381c0d578c959cc7564abf2c63612ab933bb97a55d77d21bc5c82c3ecc8b"
  );
  await linkTx.wait();
  console.log("Attestation linked to token 0!");

  const atts = await ipnft.getAttestations(0);
  console.log("Attestations on token 0:", atts.length);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
