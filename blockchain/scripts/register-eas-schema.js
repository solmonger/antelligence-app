/**
 * Register BioFVM Simulation Attestation Schema on EAS (Base Sepolia).
 *
 * Schema: SimulationResult
 * Fields:
 *   bytes32 runHash        - Unique simulation run identifier
 *   string  ipfsCid        - IPFS content ID with full simulation data
 *   bytes32 dataHash       - keccak256 of simulation output for integrity
 *   uint256 score          - Performance score (kill rate × 10000)
 *   string  strategyType   - "pheromone-guided", "rule-based", "queen-llm"
 *   uint16  nanobotCount   - Number of nanobots in simulation
 *   uint16  tumorRadius    - Tumor radius in µm
 *   uint32  steps          - Number of simulation steps
 *   bool    reproducible   - Whether attestation bot verified reproducibility
 *
 * Usage:
 *   npx hardhat run scripts/register-eas-schema.js --network baseSepolia
 */

const { ethers } = require("hardhat");
const { SchemaRegistry } = require("@ethereum-attestation-service/eas-sdk");

// Base Sepolia EAS predeploy addresses
const SCHEMA_REGISTRY_ADDRESS = "0x4200000000000000000000000000000000000020";

// BioFVM Simulation Result schema
const SCHEMA_STRING =
  "bytes32 runHash, string ipfsCid, bytes32 dataHash, uint256 score, " +
  "string strategyType, uint16 nanobotCount, uint16 tumorRadius, " +
  "uint32 steps, bool reproducible";

async function main() {
  const [signer] = await ethers.getSigners();
  console.log("Registering schema with:", signer.address);

  const schemaRegistry = new SchemaRegistry(SCHEMA_REGISTRY_ADDRESS);
  schemaRegistry.connect(signer);

  console.log("\nSchema:");
  console.log(" ", SCHEMA_STRING);

  const tx = await schemaRegistry.register({
    schema: SCHEMA_STRING,
    resolverAddress: ethers.ZeroAddress, // No resolver (open attestation)
    revocable: true,
  });

  console.log("\nWaiting for transaction...");
  const schemaUID = await tx.wait();

  console.log("\nSchema registered!");
  console.log("  Schema UID:", schemaUID);
  console.log("  View: https://base-sepolia.easscan.org/schema/view/" + schemaUID);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
