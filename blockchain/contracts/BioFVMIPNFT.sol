// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title BioFVM IP-NFT — Tokenized Simulation Configurations
/// @notice Mint NFTs representing BioFVM simulation configurations as research IP.
///         Each token links to IPFS-hosted config + metadata, and can reference
///         EAS attestation UIDs for verified results.
/// @dev Simplified from Molecule's IPNFT standard for Base L2 deployment.
///      Skips Crowdsale/Permissioner/Fractionalization for MVP.
contract BioFVMIPNFT is ERC721, ERC721URIStorage, Ownable {
    uint256 private _nextTokenId;

    /// @notice Metadata for each simulation configuration
    struct SimConfig {
        string ipfsCid;          // IPFS CID of full config YAML/JSON
        bytes32 configHash;      // keccak256 of config for integrity
        string description;      // Human-readable description
        uint32 mintedAt;         // Timestamp
        bytes32 easSchemaUID;    // EAS schema used for attestations
    }

    /// @notice Mapping from token ID to simulation config metadata
    mapping(uint256 => SimConfig) public configs;

    /// @notice Mapping from token ID to EAS attestation UIDs
    mapping(uint256 => bytes32[]) public attestations;

    event ConfigMinted(
        uint256 indexed tokenId,
        address indexed creator,
        string ipfsCid,
        bytes32 configHash,
        string description
    );

    event AttestationLinked(
        uint256 indexed tokenId,
        bytes32 attestationUID
    );

    constructor() ERC721("BioFVM Simulation IP", "BFVM-IP") Ownable(msg.sender) {}

    /// @notice Mint a new IP-NFT for a simulation configuration
    /// @param to Address to mint to (creator/researcher)
    /// @param tokenURI IPFS URI for ERC721 metadata (image, name, etc.)
    /// @param ipfsCid IPFS CID of the full simulation config
    /// @param configHash keccak256 hash of the config for integrity
    /// @param description Human-readable description of what this config models
    /// @param easSchemaUID EAS schema UID for attestations made against this config
    function mint(
        address to,
        string calldata tokenURI,
        string calldata ipfsCid,
        bytes32 configHash,
        string calldata description,
        bytes32 easSchemaUID
    ) external returns (uint256) {
        uint256 tokenId = _nextTokenId++;

        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);

        configs[tokenId] = SimConfig({
            ipfsCid: ipfsCid,
            configHash: configHash,
            description: description,
            mintedAt: uint32(block.timestamp),
            easSchemaUID: easSchemaUID
        });

        emit ConfigMinted(tokenId, to, ipfsCid, configHash, description);
        return tokenId;
    }

    /// @notice Link an EAS attestation to an IP-NFT (anyone can link)
    /// @param tokenId The IP-NFT token ID
    /// @param attestationUID The EAS attestation UID
    function linkAttestation(uint256 tokenId, bytes32 attestationUID) external {
        require(tokenId < _nextTokenId, "Token does not exist");
        attestations[tokenId].push(attestationUID);
        emit AttestationLinked(tokenId, attestationUID);
    }

    /// @notice Get all attestation UIDs linked to a token
    function getAttestations(uint256 tokenId) external view returns (bytes32[] memory) {
        return attestations[tokenId];
    }

    /// @notice Get the total number of minted configs
    function totalSupply() external view returns (uint256) {
        return _nextTokenId;
    }

    // Required overrides for ERC721URIStorage
    function tokenURI(uint256 tokenId)
        public view override(ERC721, ERC721URIStorage) returns (string memory)
    {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public view override(ERC721, ERC721URIStorage) returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
}
