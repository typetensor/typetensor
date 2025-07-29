# Security Policy

## Workflow Security

### NPM Token Protection

The NPM_TOKEN secret is protected through multiple layers:

1. **Workflow Restrictions**
   - Release workflow only runs on `push` to `main` branch
   - Explicit condition: `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`
   - Never runs on pull requests

2. **GitHub Default Protections**
   - Secrets are NOT available to workflows triggered by PRs from forks
   - First-time contributors require approval before workflows run
   - Workflow file changes in PRs cannot access secrets until merged

3. **Permission Restrictions**
   - Workflows use minimal required permissions
   - No workflow has blanket `write-all` permissions

### Repository Settings (Required)

Configure these settings in your repository:

1. **Actions Permissions** (Settings → Actions → General)
   - Set to "Allow all actions and reusable workflows"
   - Enable "Require approval for first-time contributors"
   - Enable "Require approval for all outside collaborators"

2. **Fork Pull Request Workflows** (Settings → Actions → General)
   - Set to "Require approval for fork pull request workflows"
   - Select "Require approval for all outside collaborators"

3. **Secrets Protection** (Settings → Secrets and variables → Actions)
   - Never add secrets that can be used maliciously
   - Use environment-specific secrets when possible
   - Regularly rotate tokens

4. **Branch Protection** (Settings → Branches)
   - Protect `main` branch
   - Require pull request reviews
   - Dismiss stale reviews when new commits are pushed
   - Require review from CODEOWNERS
   - Include administrators in restrictions

### Best Practices

1. **Review Workflow Changes Carefully**
   - Any PR modifying `.github/workflows/` should be scrutinized
   - Check for attempts to echo/print secrets
   - Look for suspicious third-party actions

2. **Use Environments for Production Secrets**
   ```yaml
   environment:
     name: production
     url: https://npm.com/package/@typetensor/core
   ```

3. **Audit Workflow Runs**
   - Regularly check Actions tab for unexpected runs
   - Monitor npm publishes for unauthorized releases

### What Someone CANNOT Do

Even if someone creates a malicious PR:

1. **Cannot access NPM_TOKEN** in their PR workflows
2. **Cannot trigger release workflow** from their PR
3. **Cannot modify and run workflows** with secrets in same PR
4. **Cannot bypass branch protection** without reviews

### Reporting Security Issues

If you discover a security vulnerability, please email security@typetensor.org (or create this email) instead of creating a public issue.

## NPM Package Security

1. **Two-Factor Authentication**: Enable 2FA on npm accounts
2. **Scoped Packages**: Using `@typetensor/` scope for all packages
3. **Publish Restrictions**: Only automated workflows can publish
4. **Package Provenance**: Consider enabling npm provenance