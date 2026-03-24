/**
 * Make a BioFVM simulation attestation on EAS (Base Sepolia).
 *
 * Attests a simulation result using the registered schema.
 *
 * Usage:
 *   npx hardhat run scripts/make-attestation.js --network baseSepolia
 */

const { ethers } = require("hardhat");
const { EAS, SchemaEncoder } = require("@ethereum-attestation-service/eas-sdk");

// Base Sepolia EAS predeploy
const EAS_ADDRESS = "0x4200000000000000000000000000000000000021";

// Schema UID registered earlier
const SCHEMA_UID = "0xeb32221490772f15acecbb4ab0c154877e4953ad56cf5f0932f5d9f48f9e8a33";

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Attesting with:", signer.address);

  const eas = new EAS(EAS_ADDRESS);
  eas.connect(signer);

  // Encode attestation data matching the schema
  const schemaEncoder = new SchemaEncoder(
    "bytes32 runHash, string ipfsCid, bytes32 dataHash, uint256 score, " +
    "string strategyType, uint16 nanobotCount, uint16 tumorRadius, " +
    "uint32 steps, bool reproducible"
  );

  const encodedData = schemaEncoder.encodeData([
    {
      name: "runHash",
      value: "0x7b3f4292792f0ab5b70af2dea162d486746b7a3046b4b219f66f86d19e755159",
      type: "bytes32",
    },
    {
      name: "ipfsCid",
      value: "bafkreigan5c6ndz3mncjnddloyqtzbwz72h4r2hqpa7bzef5fjqzfbydty",
      type: "string",
    },
    {
      name: "dataHash",
      value: "0x59a29ad41811195b813257747786f31021a404373c1b135c30c0b7b3b2b3d6fa",
      type: "bytes32",
    },
    { name: "score", value: 4550n, type: "uint256" },
    { name: "strategyType", value: "pheromone-guided", type: "string" },
    { name: "nanobotCount", value: 10, type: "uint16" },
    { name: "tumorRadius", value: 150, type: "uint16" },
    { name: "steps", value: 300, type: "uint32" },
    { name: "reproducible", value: true, type: "bool" },
  ]);

  console.log("\nSubmitting attestation...");

  const tx = await eas.attest({
    schema: SCHEMA_UID,
    data: {
      recipient: ethers.ZeroAddress, // No specific recipient
      expirationTime: 0n, // No expiration
      revocable: true,
      refUID: ethers.ZeroHash,
      data: encodedData,
      value: 0n,
    },
  });

  const attestationUID = await tx.wait();

  console.log("\nAttestation created!");
  console.log("  Attestation UID:", attestationUID);
  console.log("  View: https://base-sepolia.easscan.org/attestation/view/" + attestationUID);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
