#!/bin/bash
#
# NCSU Globus Configuration Helper
# Helps you find and test your NCSU endpoint configuration
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     NCSU Globus Endpoint Configuration Helper                 ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if logged into Globus
if ! globus whoami &> /dev/null; then
    echo -e "${RED}❌ Not logged into Globus CLI${NC}"
    echo ""
    echo "Please run:"
    echo "  globus login"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ Logged into Globus as:${NC} $(globus whoami)"
echo ""

# Search for NC State endpoints
echo -e "${CYAN}🔍 Searching for NC State Globus endpoints...${NC}"
echo ""

endpoints=$(globus endpoint search "NC State" --filter-scope "all" --format json 2>/dev/null)

if [ -z "$endpoints" ] || [ "$endpoints" == "[]" ]; then
    echo -e "${YELLOW}⚠️  No NC State endpoints found in search${NC}"
    echo ""
    echo "Try these searches manually:"
    echo "  globus endpoint search 'NC State Research Storage'"
    echo "  globus endpoint search 'NCSU'"
    echo "  globus endpoint search 'North Carolina State'"
    echo ""
    exit 1
fi

# Parse and display endpoints
# echo "$endpoints" | jq -r '.[] | "ID: \(.id)\nName: \(.display_name)\nOwner: \(.owner_string)\n"'

echo ""
echo -e "${CYAN}📝 Common NC State Endpoints:${NC}"
echo ""
echo "1. NC State Research Storage"
echo "   Purpose: Access /rsstu and /rs1 storage"
echo "   Path format: /rsstu/users/[group] or /rs1/researchers/[unity-id]"
echo ""
echo "2. NC State Hazel HPC Cluster"
echo "   Purpose: Access HPC compute cluster"
echo ""

# Prompt for endpoint selection
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
read -p "Enter the Endpoint ID you want to use (or 'q' to quit): " endpoint_id

if [ "$endpoint_id" == "q" ]; then
    echo "Exiting..."
    exit 0
fi

# Test the endpoint
echo ""
echo -e "${CYAN}🔍 Testing endpoint: $endpoint_id${NC}"
echo ""

# Try to list root
if globus ls "$endpoint_id:/" --format json &> /dev/null; then
    echo -e "${GREEN}✅ Successfully connected to endpoint${NC}"
    echo ""
    echo -e "${CYAN}Root directory contents:${NC}"
    globus ls "$endpoint_id:/"
    echo ""
else
    echo -e "${RED}❌ Failed to access endpoint${NC}"
    echo ""
    echo "Possible issues:"
    echo "  • Endpoint ID is incorrect"
    echo "  • You don't have permission to access this endpoint"
    echo "  • Endpoint requires authentication"
    echo ""
    exit 1
fi

# Prompt for path to semifield-upload
echo ""
read -p "Enter the path to your semifield-upload directory (e.g., /rsstu/users/group/semifield-upload): " base_path

# Test the path
echo ""
echo -e "${CYAN}🔍 Testing path: $base_path${NC}"
echo ""

if globus ls "$endpoint_id:$base_path" --format json &> /dev/null; then
    echo -e "${GREEN}✅ Successfully accessed path${NC}"
    echo ""
    echo -e "${CYAN}Directory contents:${NC}"
    globus ls "$endpoint_id:$base_path" | head -20
    echo ""
else
    echo -e "${RED}❌ Failed to access path: $base_path${NC}"
    echo ""
    echo "Try navigating step by step:"
    echo "  globus ls $endpoint_id:/"
    echo "  globus ls $endpoint_id:/rsstu"
    echo "  globus ls $endpoint_id:/rsstu/users"
    echo ""
    read -p "Would you like to browse the endpoint? (y/n): " browse
    
    if [ "$browse" == "y" ]; then
        current_path="/"
        while true; do
            echo ""
            echo -e "${CYAN}Current path: $current_path${NC}"
            echo ""
            globus ls "$endpoint_id:$current_path"
            echo ""
            read -p "Enter subdirectory name (or '..' to go up, 'q' to quit): " subdir
            
            if [ "$subdir" == "q" ]; then
                break
            elif [ "$subdir" == ".." ]; then
                current_path=$(dirname "$current_path")
            else
                current_path="$current_path/$subdir"
            fi
        done
    fi
    
    exit 1
fi

# Count batches
echo ""
echo -e "${CYAN}📊 Scanning for batches (STATE_YYYY-MM-DD format)...${NC}"
echo ""

batch_count=$(globus ls "$endpoint_id:$base_path" | grep -E '^[A-Z]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}$' | wc -l)

echo -e "${GREEN}Found $batch_count batches${NC}"
echo ""

if [ $batch_count -gt 0 ]; then
    echo "Sample batches:"
    globus ls "$endpoint_id:$base_path" | grep -E '^[A-Z]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}$' | head -10
    echo ""
fi

# Generate configuration
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Configuration complete!${NC}"
echo ""
echo -e "${CYAN}Add these lines to your globus_manager.py:${NC}"
echo ""
echo -e "${YELLOW}# In GlobusTransferManager class:"
echo "NCSU_ENDPOINT = \"$endpoint_id\""
echo "NCSU_BASE_PATH = \"$base_path\""
echo -e "${NC}"
echo ""

# Save to file
config_file="/tmp/ncsu_globus_config.txt"
cat > "$config_file" << EOF
# NCSU Globus Configuration
# Generated: $(date)
# User: $(globus whoami)

NCSU_ENDPOINT = "$endpoint_id"
NCSU_BASE_PATH = "$base_path"

# Batches found: $batch_count

# To apply:
# Edit ~/repos/svs-raw-api/scripts/globus_manager.py
# Update the NCSU_ENDPOINT and NCSU_BASE_PATH constants
EOF

echo -e "${GREEN}✅ Configuration saved to: $config_file${NC}"
echo ""

# Offer to test comparison with JUNO
echo ""
read -p "Would you like to test comparison with JUNO? (y/n): " test_comparison

if [ "$test_comparison" == "y" ]; then
    echo ""
    echo -e "${CYAN}🔍 Comparing with JUNO...${NC}"
    echo ""
    
    JUNO_ENDPOINT="904c2108-90cf-11e8-9672-0a6d4e044368"
    JUNO_PATH="/project/dash_agir/semifield-upload"
    
    echo "NCSU batches:"
    ncsu_batches=$(globus ls "$endpoint_id:$base_path" | grep -E '^[A-Z]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort)
    echo "$ncsu_batches"
    
    echo ""
    echo "JUNO batches:"
    juno_batches=$(globus ls "$JUNO_ENDPOINT:$JUNO_PATH" | grep -E '^[A-Z]{2}_[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort)
    echo "$juno_batches"
    
    echo ""
    echo -e "${CYAN}Missing in JUNO:${NC}"
    comm -23 <(echo "$ncsu_batches") <(echo "$juno_batches")
    
    echo ""
    echo -e "${CYAN}Missing in NCSU:${NC}"
    comm -13 <(echo "$ncsu_batches") <(echo "$juno_batches")
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Configuration complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Edit ~/repos/svs-raw-api/scripts/globus_manager.py"
echo "  2. Update NCSU_ENDPOINT and NCSU_BASE_PATH with values above"
echo "  3. Test: ./scripts/workflow.sh check-missing"
echo ""
