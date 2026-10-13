// Deploy the Antelligence v4.1 contracts (AnchorRegistry, BoardLog, ReputationRegistry) to ZKsync.
//
//   V41_DRY_RUN=1 npx hardhat run scripts/deploy-zksync-v41.js --network inMemoryNode          # estimate only
//   npx hardhat run scripts/deploy-zksync-v41.js --network inMemoryNode                        # local rehearsal (chain 260)
//   npx hardhat run scripts/deploy-zksync-v41.js --network zkSyncSepoliaTestnet                # testnet (chain 300)
//
// Same safety checks as deploy-zksync.js: only chains 260 and 300; the deployer balance must cover the
// estimated cost x1.5; the record carries commit, dirty-tree flag and compiler versions.
// Dry-run mode (V41_DRY_RUN=1) estimates gas, prints the preflight and exits: no transaction, no file write.
//
// The result is written under a new `v4_1` key of blockchain/deployments/<network>.json. Every other key
// (the original contracts list, gas, compiler, ...) is preserved byte-for-byte in meaning. An existing
// `v4_1` record is never overwritten unless V41_REDEPLOY=1.
//
// Helpers are duplicated from deploy-zksync.js on purpose: that file runs main() on require, and `hardhat run`
// loads scripts with require, so importing it would trigger a v1 deploy.

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const hre = require("hardhat");
const { Wallet, Provider } = require("zksync-ethers");
const { Deployer } = require("@matterlabs/hardhat-zksync-deploy");

const ALLOWED = {
  260: { name: "zksync-local-inmemory", explorer: null, committed: false },
  300: { name: "zksync-era-sepolia", explorer: "https://sepolia.explorer.zksync.io", committed: true },
};
const CONTRACTS = [
  { name: "AnchorRegistry", args: () => [] },
  { name: "BoardLog", args: () => [] },
  { name: "ReputationRegistry", args: () => [] },
];
const SAFETY_FACTOR_NUM = 3n; // balance must be >= estimate * 1.5
const SAFETY_FACTOR_DEN = 2n;
const DRY_RUN = ["1", "true", "yes"].includes(String(process.env.V41_DRY_RUN || "").toLowerCase());
const REDEPLOY = process.env.V41_REDEPLOY === "1";

function git(cmd) {
  return execSync(`git ${cmd}`, { cwd: path.join(__dirname, ".."), encoding: "utf8" }).trim();
}

function compilerVersions() {
  const buildInfoDir = path.join(__dirname, "..", "artifacts-zk", "build-info");
  const file = fs.readdirSync(buildInfoDir).filter((f) => f.endsWith(".json"))[0];
  const info = JSON.parse(fs.readFileSync(path.join(buildInfoDir, file), "utf8"));
  const settings = hre.config.solidity.compilers[0].settings;
  return {
    zksolc: hre.config.zksolc.version,
    solc: info.solcLongVersion,
    zksolc_settings: hre.config.zksolc.settings,
    solc_optimizer: settings.optimizer,
    via_ir: settings.viaIR,
  };
}

function deployerWallet(provider, chainId) {
  if (chainId === 260) {
    // Public, well-known anvil-zksync dev account #0 (local node only; holds no real value).
    const { richWallets } = require("@matterlabs/hardhat-zksync-ethers/dist/rich-wallets");
    return new Wallet(richWallets["0x104"][0].privateKey, provider);
  }
  const accounts = hre.network.config.accounts;
  if (!Array.isArray(accounts) || accounts.length === 0) {
    throw new Error("No deployer key configured for this network (PRIVATE_KEY in ../.env).");
  }
  return new Wallet(accounts[0], provider);
}

async function main() {
  if (!hre.network.config.zksync) throw new Error(`Network ${hre.network.name} is not a ZKsync network.`);
  const provider = new Provider(hre.network.config.url);
  const chainId = Number((await provider.getNetwork()).chainId);
  const target = ALLOWED[chainId];
  if (!target) throw new Error(`Refusing to deploy: chain ${chainId} is not an allowed testnet/local chain.`);

  const dir = path.join(__dirname, "..", "deployments");
  const file = path.join(dir, `${target.name}.json`);
  const existing = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : null;
  if (existing && existing.chain_id !== undefined && existing.chain_id !== chainId) {
    throw new Error(`${path.basename(file)} records chain ${existing.chain_id}, connected chain is ${chainId}. Not deploying.`);
  }
  if (existing && existing.v4_1 && !REDEPLOY && !DRY_RUN) {
    throw new Error(`${path.basename(file)} already has a v4_1 record. Set V41_REDEPLOY=1 to replace it deliberately.`);
  }

  const wallet = deployerWallet(provider, chainId);
  const deployer = new Deployer(hre, wallet);
  const balance = await provider.getBalance(wallet.address);

  // Pre-flight: estimate every deploy before sending any transaction.
  const artifacts = {};
  const perContract = {};
  let estimate = 0n;
  for (const c of CONTRACTS) {
    artifacts[c.name] = await deployer.loadArtifact(c.name);
    const fee = await deployer.estimateDeployFee(artifacts[c.name], c.args(wallet.address));
    perContract[c.name] = fee.toString();
    estimate += fee;
  }
  const needed = (estimate * SAFETY_FACTOR_NUM) / SAFETY_FACTOR_DEN;
  const preflight = {
    mode: DRY_RUN ? "dry-run" : "deploy",
    chain_id: chainId,
    network: hre.network.name,
    deployer: wallet.address,
    balance_wei: balance.toString(),
    estimated_deploy_fee_wei: estimate.toString(),
    estimated_per_contract_wei: perContract,
    required_with_margin_wei: needed.toString(),
    balance_covers_margin: balance >= needed,
  };
  console.log(JSON.stringify({ preflight }, null, 2));
  if (DRY_RUN) {
    console.log("Dry run: no transaction sent, no file written.");
    return;
  }
  if (balance < needed) {
    throw new Error(
      `Insufficient testnet gas: balance ${balance} wei < required ${needed} wei (estimate x1.5). Not deploying.`,
    );
  }

  const commit = git("rev-parse HEAD");
  const dirty = git("status --porcelain -- contracts hardhat.config.js scripts") !== "";
  const records = [];
  for (const c of CONTRACTS) {
    const contract = await deployer.deploy(artifacts[c.name], c.args(wallet.address));
    const receipt = await contract.deploymentTransaction().wait();
    const address = await contract.getAddress();
    records.push({
      name: c.name,
      address,
      deploy_tx_hash: receipt.hash,
      block_number: receipt.blockNumber,
      deployer: wallet.address,
      is_mock: false,
      constructor_args: c.args(wallet.address),
      gas_used: receipt.gasUsed.toString(),
      explorer_address_url: target.explorer ? `${target.explorer}/address/${address}` : null,
      explorer_tx_url: target.explorer ? `${target.explorer}/tx/${receipt.hash}` : null,
    });
    console.log(`${c.name} -> ${address} (tx ${receipt.hash}, block ${receipt.blockNumber})`);
  }
  const balanceAfter = await provider.getBalance(wallet.address);

  const v41 = {
    schema: "antelligence.deployment.v4_1/v1",
    deployer: wallet.address,
    deployed_at_utc: new Date().toISOString(),
    git_commit: commit,
    git_tree_dirty_for_blockchain_dir: dirty,
    compiler: compilerVersions(),
    gas: {
      balance_before_wei: balance.toString(),
      balance_after_wei: balanceAfter.toString(),
      spent_wei: (balance - balanceAfter).toString(),
      estimated_deploy_fee_wei: estimate.toString(),
    },
    contracts: records,
  };
  // Re-read just before writing so a concurrent edit of the other keys is not clobbered.
  const base = fs.existsSync(file)
    ? JSON.parse(fs.readFileSync(file, "utf8"))
    : { schema: "antelligence.deployment/v1", network: target.name, chain_id: chainId, vm: "EraVM" };
  base.v4_1 = v41;
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(file, JSON.stringify(base, null, 2) + "\n");
  console.log(`Wrote v4_1 key of ${path.relative(path.join(__dirname, ".."), file)}`);
}

main().catch((err) => {
  console.error(err.message || err);
  process.exitCode = 1;
});
