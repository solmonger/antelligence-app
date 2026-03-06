// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TumorIntel
 * @dev Smart contract for nanobots to share intelligence on the tumor battlefield.
 * Enables decentralized, verifiable communication between autonomous agents.
 */
contract TumorIntel {
    enum PinType { 
        HYPOXIC_CLUSTER,      // Area with low oxygen - high priority target
        STEM_CELL_DETECTED,   // Cancer stem cell found - requires special attention
        HIGH_RESISTANCE_AREA, // Area where drug resistance is high
        VESSEL_LOCATION,      // Blood vessel for reloading detected
        SUCCESSFUL_KILL,      // Cell successfully eliminated
        DRUG_OVERDOSE_ZONE,   // Area with too much drug concentration
        TARGET_ACQUIRED,      // Nanobot has acquired a tumor cell target
        DRUG_DELIVERY         // Drug delivered to target location
    }

    struct IntelPin {
        uint x;               // X coordinate in micrometers
        uint y;               // Y coordinate in micrometers
        PinType pinType;      // Type of intelligence
        address reporter;     // Nanobot that reported this
        uint timestamp;       // When this was reported
        uint priority;        // Priority level (1-10)
        bool isActive;        // Whether this intel is still relevant
    }

    IntelPin[] public intelPins;
    
    // Mapping from pin ID to confirmation count (how many nanobots confirmed this)
    mapping(uint => uint) public confirmations;
    
    // Mapping to track which nanobot confirmed which pin
    mapping(uint => mapping(address => bool)) public hasConfirmed;

    event IntelReported(
        uint indexed pinId,
        uint x,
        uint y,
        PinType pinType,
        address reporter,
        uint priority
    );

    event IntelConfirmed(
        uint indexed pinId,
        address confirmer,
        uint totalConfirmations
    );

    event IntelDeactivated(
        uint indexed pinId,
        address deactivator
    );

    event IntelPriorityUpdated(
        uint indexed pinId,
        uint oldPriority,
        uint newPriority,
        address updater
    );

    /**
     * @dev Report new intelligence to the battlefield
     * @param x X coordinate in micrometers
     * @param y Y coordinate in micrometers
     * @param pinType Type of intelligence being reported
     * @param priority Priority level (1-10, 10 being highest)
     */
    function reportIntel(uint x, uint y, PinType pinType, uint priority) public returns (uint) {
        require(priority >= 1 && priority <= 10, "Priority must be between 1 and 10");
        
        uint pinId = intelPins.length;
        intelPins.push(IntelPin({
            x: x,
            y: y,
            pinType: pinType,
            reporter: msg.sender,
            timestamp: block.timestamp,
            priority: priority,
            isActive: true
        }));
        
        emit IntelReported(pinId, x, y, pinType, msg.sender, priority);
        return pinId;
    }

    /**
     * @dev Confirm an existing intel report (increases its reliability)
     * @param pinId ID of the intel pin to confirm
     */
    function confirmIntel(uint pinId) public {
        require(pinId < intelPins.length, "Invalid pin ID");
        require(intelPins[pinId].isActive, "Intel is no longer active");
        require(!hasConfirmed[pinId][msg.sender], "Already confirmed this intel");
        
        hasConfirmed[pinId][msg.sender] = true;
        confirmations[pinId] += 1;
        
        emit IntelConfirmed(pinId, msg.sender, confirmations[pinId]);
    }

    /**
     * @dev Deactivate an intel report (e.g., when the situation has changed)
     * @param pinId ID of the intel pin to deactivate
     */
    function deactivateIntel(uint pinId) public {
        require(pinId < intelPins.length, "Invalid pin ID");
        require(intelPins[pinId].isActive, "Intel already deactivated");
        
        // Only the original reporter or someone who confirmed can deactivate
        require(
            msg.sender == intelPins[pinId].reporter || hasConfirmed[pinId][msg.sender],
            "Not authorized to deactivate"
        );
        
        intelPins[pinId].isActive = false;
        
        emit IntelDeactivated(pinId, msg.sender);
    }

    /**
     * @dev Update the priority of an existing intel report
     * @param pinId ID of the intel pin to update
     * @param newPriority New priority level (1-10)
     */
    function updateIntelPriority(uint pinId, uint newPriority) public {
        require(pinId < intelPins.length, "Invalid pin ID");
        require(intelPins[pinId].isActive, "Intel is no longer active");
        require(newPriority >= 1 && newPriority <= 10, "Priority must be between 1 and 10");
        require(
            msg.sender == intelPins[pinId].reporter || hasConfirmed[pinId][msg.sender],
            "Not authorized to update priority"
        );
        
        uint oldPriority = intelPins[pinId].priority;
        intelPins[pinId].priority = newPriority;
        
        emit IntelPriorityUpdated(pinId, oldPriority, newPriority, msg.sender);
    }

    /**
     * @dev Get the total number of intel pins
     * @return count Total number of pins
     */
    function getPinCount() public view returns (uint) {
        return intelPins.length;
    }

    /**
     * @dev Get all active intel pins
     * @return activePins Array of active pin IDs
     */
    function getActivePins() public view returns (uint[] memory) {
        uint activeCount = 0;
        
        // First, count active pins
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].isActive) {
                activeCount++;
            }
        }
        
        // Then, collect active pin IDs
        uint[] memory activePins = new uint[](activeCount);
        uint index = 0;
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].isActive) {
                activePins[index] = i;
                index++;
            }
        }
        
        return activePins;
    }

    /**
     * @dev Get intel pins by type
     * @param pinType Type of intel to filter by
     * @return pinIds Array of pin IDs matching the type
     */
    function getPinsByType(PinType pinType) public view returns (uint[] memory) {
        uint count = 0;
        
        // First, count matching pins
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].pinType == pinType && intelPins[i].isActive) {
                count++;
            }
        }
        
        // Then, collect matching pin IDs
        uint[] memory pinIds = new uint[](count);
        uint index = 0;
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].pinType == pinType && intelPins[i].isActive) {
                pinIds[index] = i;
                index++;
            }
        }
        
        return pinIds;
    }

    /**
     * @dev Get high priority intel pins (priority >= 8)
     * @return highPriorityPins Array of high priority pin IDs
     */
    function getHighPriorityPins() public view returns (uint[] memory) {
        uint count = 0;
        
        // First, count high priority pins
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].priority >= 8 && intelPins[i].isActive) {
                count++;
            }
        }
        
        // Then, collect high priority pin IDs
        uint[] memory highPriorityPins = new uint[](count);
        uint index = 0;
        for (uint i = 0; i < intelPins.length; i++) {
            if (intelPins[i].priority >= 8 && intelPins[i].isActive) {
                highPriorityPins[index] = i;
                index++;
            }
        }
        
        return highPriorityPins;
    }

    /**
     * @dev Get intel pins within a specific area
     * @param centerX X coordinate of area center
     * @param centerY Y coordinate of area center
     * @param radius Search radius in micrometers
     * @return pinIds Array of pin IDs within the area
     */
    function getPinsInArea(uint centerX, uint centerY, uint radius) public view returns (uint[] memory) {
        uint count = 0;
        uint radiusSquared = radius * radius;
        
        // First, count pins within radius
        for (uint i = 0; i < intelPins.length; i++) {
            if (!intelPins[i].isActive) continue;
            
            int deltaX = int(intelPins[i].x) - int(centerX);
            int deltaY = int(intelPins[i].y) - int(centerY);
            uint distanceSquared = uint(deltaX * deltaX + deltaY * deltaY);
            
            if (distanceSquared <= radiusSquared) {
                count++;
            }
        }
        
        // Then, collect pin IDs within radius
        uint[] memory pinIds = new uint[](count);
        uint index = 0;
        for (uint i = 0; i < intelPins.length; i++) {
            if (!intelPins[i].isActive) continue;
            
            int deltaX = int(intelPins[i].x) - int(centerX);
            int deltaY = int(intelPins[i].y) - int(centerY);
            uint distanceSquared = uint(deltaX * deltaX + deltaY * deltaY);
            
            if (distanceSquared <= radiusSquared) {
                pinIds[index] = i;
                index++;
            }
        }
        
        return pinIds;
    }
}