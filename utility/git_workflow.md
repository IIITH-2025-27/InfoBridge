# Git Workflow Guide

## 🔹 0. Prerequisites & Initial Setup

Before you start working, ensure you have:
- Git installed on your system
- Access to the repository on GitHub
- Your branch should already be created (or create it following step 1)

### ✔ Clone the Repository (First Time Only)
```bash
git clone <repository-url>
cd InfoBridge
```

### ✔ Check Your Current Branch
```bash
git branch -a              # List all branches (local and remote)
git branch                 # List only local branches
git status                 # Shows current branch and changes
```

---

## 🔹 1. Branching Rule

Each team member must work on their own branch, named exactly after their name.

### ✔ Branch Creation
```bash
git checkout -b <your-name>
# example:
git checkout -b vikash
```

## 🔹 2. Always Sync Before You Start Working

Before writing any code, sync your personal branch with the latest main branch.

```bash
git checkout main
git pull origin main
git checkout <your-name>
git merge main
```

This brings all the latest changes from main into your branch, ensuring you're working on up-to-date code.

## 🔹 3. Do All Work in Your Own Branch

Make changes, test locally, and commit inside your personal branch.

```bash
git status
git add .
git commit -m "Meaningful commit message"
```

## 🔹 4. Push Your Branch to Remote

Push your branch so the team can review your work.

```bash
git push origin <your-name>
```

## 🔹 5. Merge to main Only After Testing

Your branch can be merged to main ONLY when:

- All changes are tested
- Code is stable
- No conflicts

### ✔ Merge Steps (via Pull Request)

1. Push your branch to remote:
   ```bash
   git push origin <your-name>
   ```
2. Go to your repository on GitHub

3. Click on "Pull Requests" -> "New Pull Request"

4. Select:
   - Base branch: `main`
   - Compare branch: `<your-name>`

5. Add a descriptive title and description of your changes

6. Click "Merge Pull Request" once approved

## � 6. After a PR is Merged

Once any team member's PR is merged to main, **everyone** should sync their branches:

```bash
git checkout main
git pull origin main
git checkout <your-name>
git merge main
```

This keeps all branches up-to-date and reduces future conflicts.

## 🔹 7. Handling Merge Conflicts

If you get conflicts when merging main into your branch:

```bash
# After running: git merge main
# If conflicts occur:
git status                    # See conflicted files
# Open each file and resolve conflicts manually
git add <resolved-file>
git commit -m "Resolved merge conflicts"
```

### Tips:
- Look for `<<<<<<<`, `=======`, `>>>>>>>` markers in files
- Keep the code you want, delete the markers
- Test your code after resolving

## 🔹 8. Best Practices for Commit Messages

Write clear, descriptive commit messages that explain WHAT changed and WHY:

**Good Examples:**
```bash
git commit -m "Add user authentication module"
git commit -m "Fix bug in data retrieval endpoint"
git commit -m "Refactor database connection handling"
git commit -m "Update requirements.txt with new dependencies"
```

**Bad Examples:**
```bash
git commit -m "changes"
git commit -m "update"
git commit -m "fix"
```

### Tips:
- Use imperative mood: "Add" not "Added"
- Keep subject line under 50 characters
- If needed, add detailed description on separate lines
- Reference issue numbers if applicable: "Fix #123"

---

## 🔹 9. Undoing Changes Before Pushing

If you made commits you want to undo (before pushing):

```bash
# View recent commits
git log --oneline -n 5

# Undo last commit (keep changes unstaged)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo changes in a specific file
git checkout -- <filename>
```

**WARNING:** Don't use `git reset --hard` if you're unsure - it permanently deletes changes!

---

## 🔹 10. Clean Up After Merge

Once your PR is merged to main, clean up your local and remote branches:

```bash
# Delete local branch
git branch -d <your-name>

# Delete remote branch (optional, can be done on GitHub)
git push origin --delete <your-name>
```

If deletion fails with "not fully merged" warning:
```bash
git branch -D <your-name>  # Force delete (use carefully)
```

---

## 🔹 11. Stashing Work in Progress

If you need to switch branches but have uncommitted changes:

```bash
# Save current work temporarily
git stash

# Switch to another branch
git checkout main

# Come back and restore your work
git checkout <your-name>
git stash pop
```

---

## 🔹 12. Pulling Recent Changes from Your Branch

If your branch is pushed and you're working from multiple machines:

```bash
git fetch origin
git pull origin <your-name>
```

---

## 🔹 13. Important Rules

### ✅ DO's
- ✅ Always sync your branch before starting work
- ✅ Sync after any PR is merged to main
- ✅ Write clear, meaningful commit messages
- ✅ Test your code locally before pushing
- ✅ Create a pull request for code review
- ✅ Keep your commits focused and atomic
- ✅ Pull latest changes from your branch if working on multiple machines

### ❌ DON'Ts
- ❌ Never work directly on main (except for Abhay)
- ❌ Never merge unfinished work
- ❌ No force pushes on main or shared branches
- ❌ Don't commit without meaningful messages
- ❌ Don't push directly to main
- ❌ Don't ignore merge conflicts
- ❌ Don't leave stale branches without cleaning up

---

## 🔹 14. Troubleshooting

### Accidentally committed to main?
```bash
git log --oneline -n 3              # Find your commits
git checkout -b <your-name>         # Create your branch
git checkout main
git reset --hard <commit-before-yours>  # Move main back
```

### Branch is too far behind main?
```bash
git checkout <your-name>
git fetch origin
git rebase origin/main              # More linear history
# or
git merge origin/main               # Merge commit approach
```

### Need to see what changed?
```bash
git diff                            # Unstaged changes
git diff --staged                   # Staged changes
git diff main <your-name>           # Diff with main
```

---

## 📚 Quick Command Reference

| Task | Command |
|------|---------|
| Create your branch | `git checkout -b <your-name>` |
| Switch to branch | `git checkout <your-name>` |
| View current branch | `git status` or `git branch` |
| Sync with main | `git fetch origin && git merge origin/main` |
| Stage changes | `git add .` |
| Commit changes | `git commit -m "message"` |
| Push to remote | `git push origin <your-name>` |
| View commits | `git log --oneline` |
| Undo last commit | `git reset --soft HEAD~1` |
| Stash work | `git stash` |
| Restore stashed work | `git stash pop` |

---
