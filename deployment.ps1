# PowerShell script for deployment to Harbor and DB migration
# Usage: .\deployment.ps1 -Version "0.1.37" [-RunMigration] [-SkipBuild] [-SkipPush]

param(
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [Parameter(Mandatory=$false)]
    [switch]$RunMigration = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipBuild = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipPush = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$HarborRegistry = "harbor.agrichain.com.ua",
    
    [Parameter(Mandatory=$false)]
    [string]$HarborProject = "agrichainserver",
    
    [Parameter(Mandatory=$false)]
    [string]$ImageName = "ai-sugar-inclusions"
)

$ErrorActionPreference = "Stop"

# Helper functions for colored output
function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Full image name
$LocalImageName = $ImageName
$HarborImageName = "$HarborRegistry/$HarborProject/${ImageName}:${Version}"

Write-Host "`n========================================================" -ForegroundColor Magenta
Write-Host "       Sugar Inclusions Classifier - DEPLOYMENT        " -ForegroundColor Magenta
Write-Host "========================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Version:     $Version" -ForegroundColor White
Write-Host "Image:       $HarborImageName" -ForegroundColor White
$migrationStatus = if ($RunMigration) { 'YES' } else { 'NO' }
Write-Host "Migration:   $migrationStatus" -ForegroundColor White
Write-Host ""

# ===================================
# 1. BUILD DOCKER IMAGE
# ===================================
if (-not $SkipBuild) {
    Write-Step "STEP 1: Building Docker Image"
    
    try {
        Write-Host "Building image: $LocalImageName" -ForegroundColor Yellow
        docker build -t $LocalImageName .
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed with exit code $LASTEXITCODE"
        }
        
        Write-Success "Docker image built successfully: $LocalImageName"
        
        # Get image info
        $imageInfo = docker images $LocalImageName --format '{{.Size}}'
        Write-Host "Image size: $imageInfo" -ForegroundColor Gray
        
    } catch {
        Write-Error-Custom "Failed to build Docker image: $_"
        exit 1
    }
} else {
    Write-Warning-Custom "Skipping Docker build (using existing image)"
}

# ===================================
# 2. TAG IMAGE FOR HARBOR
# ===================================
Write-Step "STEP 2: Tagging Image for Harbor"

try {
    Write-Host "Tagging: $LocalImageName -> $HarborImageName" -ForegroundColor Yellow
    docker tag $LocalImageName $HarborImageName
    
    if ($LASTEXITCODE -ne 0) {
        throw "Docker tag failed with exit code $LASTEXITCODE"
    }
    
    Write-Success "Image tagged: $HarborImageName"
    
} catch {
    Write-Error-Custom "Failed to tag Docker image: $_"
    exit 1
}

# ===================================
# 3. LOGIN TO HARBOR REGISTRY
# ===================================
if (-not $SkipPush) {
    Write-Step "STEP 3: Login to Harbor Registry"
    
    try {
        Write-Host "Logging in to: $HarborRegistry" -ForegroundColor Yellow
        Write-Host "Please enter your Harbor credentials:" -ForegroundColor Yellow
        
        docker login $HarborRegistry
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker login failed with exit code $LASTEXITCODE"
        }
        
        Write-Success "Successfully logged in to Harbor"
        
    } catch {
        Write-Error-Custom "Failed to login to Harbor: $_"
        exit 1
    }
}

# ===================================
# 4. PUSH IMAGE TO HARBOR
# ===================================
if (-not $SkipPush) {
    Write-Step "STEP 4: Pushing Image to Harbor"
    
    try {
        Write-Host "Pushing: $HarborImageName" -ForegroundColor Yellow
        docker push $HarborImageName
        
        if ($LASTEXITCODE -ne 0) {
            throw "Docker push failed with exit code $LASTEXITCODE"
        }
        
        Write-Success "Image pushed successfully to Harbor"
        Write-Host "Registry URL: https://$HarborRegistry" -ForegroundColor Gray
        
    } catch {
        Write-Error-Custom "Failed to push Docker image: $_"
        exit 1
    }
} else {
    Write-Warning-Custom "Skipping Docker push"
}

# ===================================
# 5. DATABASE MIGRATION
# ===================================
if ($RunMigration) {
    Write-Step "STEP 5: Running Database Migration"
    
    try {
        # Check if alembic.ini exists
        if (-not (Test-Path "src/migrations/alembic.ini")) {
            throw "alembic.ini not found in src/migrations/"
        }
        
        # Activate venv if exists
        if (Test-Path "venv\Scripts\Activate.ps1") {
            Write-Host "Activating virtual environment..." -ForegroundColor Yellow
            & "venv\Scripts\Activate.ps1"
        }
        
        # Check if alembic is installed
        $alembicCheck = python -m pip list | Select-String "alembic"
        if (-not $alembicCheck) {
            Write-Warning-Custom "Alembic not found, installing..."
            python -m pip install alembic
        }
        
        Write-Host "`nRunning Alembic migrations..." -ForegroundColor Yellow
        Write-Host "Configuration: src/migrations/alembic.ini" -ForegroundColor Gray
        
        # Show current DB version
        Write-Host "`nCurrent database version:" -ForegroundColor Yellow
        alembic -c src/migrations/alembic.ini current
        
        # Show migration history
        Write-Host "`nMigration history:" -ForegroundColor Yellow
        alembic -c src/migrations/alembic.ini history
        
        # Run migration
        Write-Host "`nUpgrading to latest version..." -ForegroundColor Yellow
        alembic -c src/migrations/alembic.ini -x echo=true upgrade head
        
        if ($LASTEXITCODE -ne 0) {
            throw "Alembic migration failed with exit code $LASTEXITCODE"
        }
        
        # Show new version
        Write-Host "`nNew database version:" -ForegroundColor Yellow
        alembic -c src/migrations/alembic.ini current
        
        Write-Success "Database migration completed successfully"
        
    } catch {
        Write-Error-Custom "Failed to run database migration: $_"
        Write-Host "`nTo create a new migration, run:" -ForegroundColor Yellow
        Write-Host '  alembic -c src/migrations/alembic.ini revision --autogenerate -m "Description"' -ForegroundColor Gray
        exit 1
    }
} else {
    Write-Warning-Custom "Skipping database migration (use -RunMigration flag to enable)"
}

# ===================================
# 6. SUMMARY
# ===================================
Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "                  DEPLOYMENT COMPLETED                  " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Image deployed:  $HarborImageName" -ForegroundColor White
Write-Host "Registry:        https://$HarborRegistry" -ForegroundColor White
Write-Host ""

if ($RunMigration) {
    Write-Host "[OK] Database migration: COMPLETED" -ForegroundColor Green
} else {
    Write-Host "[ ] Database migration: SKIPPED" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Update Kubernetes deployment to use: $HarborImageName" -ForegroundColor Gray
Write-Host "  2. Run: kubectl set image deployment/sugar-inclusions ai-sugar-inclusions=$HarborImageName" -ForegroundColor Gray
Write-Host "  3. Monitor: kubectl rollout status deployment/sugar-inclusions" -ForegroundColor Gray
Write-Host ""

# Additional information
Write-Host "Commands reference:" -ForegroundColor Cyan
Write-Host "  # View logs:" -ForegroundColor Gray
Write-Host "    kubectl logs -f deployment/sugar-inclusions" -ForegroundColor DarkGray
Write-Host "  # Restart deployment:" -ForegroundColor Gray
Write-Host "    kubectl rollout restart deployment/sugar-inclusions" -ForegroundColor DarkGray
Write-Host "  # Delete old image:" -ForegroundColor Gray
Write-Host "    docker rmi $HarborRegistry/$HarborProject/${ImageName}:<old-version>" -ForegroundColor DarkGray
Write-Host ""

Write-Success "Deployment script completed!"
