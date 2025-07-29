# Release Process

This document explains how releases are automated for TypeTensor using GitHub Actions and Changesets.

## Overview

TypeTensor uses [Changesets](https://github.com/changesets/changesets) to manage versions and releases. The process is fully automated:

1. Developers create changesets for their changes
2. A GitHub Action creates/updates a release PR
3. Merging the release PR publishes to npm

## How It Works

### 1. Pull Request Requirements

All PRs that modify package code must include a changeset. This is enforced by CI checks that will:
- Detect changes to files in `packages/` directory
- Verify a changeset exists (excluding README.md)
- Block merging if changeset is missing
- Post helpful comments on how to add changesets

Exceptions:
- Changes to test files (*.test.ts, *.spec.ts)
- Changes to type definition files (*.d.ts)
- Documentation-only changes
- Configuration file updates

### 2. Creating Changesets

When you make changes that should be released, create a changeset:

```bash
bun changeset
```

This will prompt you to:
- Select which packages have changed
- Choose the version bump type (patch/minor/major)
- Write a summary of the changes

The changeset is saved in `.changeset/` directory.

**Empty Changesets**: If your changes don't require a release (internal refactoring, tests), create an empty changeset:
```bash
bun changeset --empty
```

### 3. Automated Release PR

When changesets are pushed to `main`, the release workflow:

1. Detects all changesets
2. Creates or updates a "Release" PR that:
   - Bumps package versions according to changesets
   - Updates CHANGELOG.md files
   - Removes processed changesets

### 4. Publishing to npm

When you merge the Release PR:

1. The workflow builds all packages
2. Publishes updated packages to npm
3. Creates git tags for each release
4. Updates GitHub releases

## Version Bump Guidelines

### Patch Release (0.0.X)
- Bug fixes
- Performance improvements
- Documentation updates
- Internal refactoring

### Minor Release (0.X.0)
- New features (backward compatible)
- New APIs
- Deprecations (with migration path)

### Major Release (X.0.0)
- Breaking changes
- Removed deprecated features
- Major architectural changes

## Setup Requirements

### 1. NPM Token

You need to add an NPM_TOKEN secret to your GitHub repository:

1. Create an npm access token:
   ```bash
   npm login
   npm token create --read-only=false
   ```

2. Add to GitHub:
   - Go to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `NPM_TOKEN`
   - Value: Your npm token

### 2. GitHub Token

The default `GITHUB_TOKEN` is used automatically and requires:
- Write access to contents (for version bumps)
- Write access to pull requests (for creating release PRs)

## Manual Release Process

If needed, you can release manually:

```bash
# Create a changeset
bun changeset

# Version packages
bun run version

# Build and publish
bun run release
```

## Workflow Configuration

The release workflow (`.github/workflows/release.yml`) runs on every push to `main` and:

1. Uses `changesets/action` to manage the release process
2. Creates a PR when there are changesets
3. Publishes when the PR is merged

## Troubleshooting

### Release PR not created
- Check if changesets exist in `.changeset/`
- Verify the workflow has proper permissions
- Check workflow runs in Actions tab

### Publishing failed
- Verify NPM_TOKEN is set correctly
- Check npm authentication
- Ensure package names are available
- Verify `access: "public"` in changeset config

### Version conflicts
- Changesets handles version bumps automatically
- For complex scenarios, use `fixed` or `linked` versioning in config

## Best Practices

1. **Always create changesets** for user-facing changes
2. **Write clear changeset summaries** - they become changelog entries
3. **Use semantic versioning** correctly
4. **Review the release PR** before merging
5. **Test locally** with `bun run build` before pushing

## Example Changeset

```markdown
---
"@typetensor/core": minor
"@typetensor/backend-cpu": patch
---

Added new reshape operations with compile-time shape validation.
Fixed memory leak in CPU backend tensor disposal.
```

This creates a minor bump for core and patch for backend-cpu.