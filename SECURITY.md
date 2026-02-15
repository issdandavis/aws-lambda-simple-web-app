# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to: aethermoregames@pm.me
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

### Response Timeline

| Severity | Initial Response | Resolution Target |
|----------|------------------|-------------------|
| Critical | 24 hours | 7 days |
| High | 48 hours | 14 days |
| Medium | 7 days | 30 days |
| Low | 14 days | 90 days |

### What to Expect

1. Acknowledgment of your report within the response time
2. Regular updates on our progress
3. Credit in the security advisory (unless you prefer anonymity)
4. Notification when the vulnerability is fixed

## Security Best Practices

### For Developers

1. **No secrets in code** - Use environment variables or secret managers
2. **Pin dependencies** - Use lockfiles with hashes
3. **Run security scans** - Bandit for Python, npm audit for Node
4. **Review PRs** - All changes require review

### SCBE-AETHERMOORE Security

This project implements the Symphonic Cipher / SCBE-AETHERMOORE security framework with:

| Purpose | Algorithm | Standard |
|---------|-----------|----------|
| Key Encapsulation | ML-KEM-768 | NIST FIPS 203 |
| Digital Signatures | ML-DSA-65 | NIST FIPS 204 |
| Harmonic Scaling | H(d) = exp(d^2) | SCBE Axiom A12 |

## Security Contacts

- Email: aethermoregames@pm.me
- GitHub Security Advisories: Configure in repo settings
