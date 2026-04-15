# Git Workflow Guide

## 🔹 1. Branching Rule

Each team member must work on their own branch, named exactly after their name.

### ✔ Branch Creation
```bash
git checkout -b <your-name>
# example:
git checkout -b vikash
```

## 🔹 2. Always Sync Before You Start Working

Before writing any code, sync your personal branch with the latest master branch.

```bash
git checkout master
git pull origin master
git checkout <your-name>
git merge master
```

This brings all the latest changes from master into your branch, ensuring you're working on up-to-date code.

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

## 🔹 5. Merge to Master Only After Testing

Your branch can be merged to master ONLY when:

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
   - Base branch: `master`
   - Compare branch: `<your-name>`

5. Add a descriptive title and description of your changes

6. Click "Merge Pull Request" once approved

## � 6. After a PR is Merged

Once any team member's PR is merged to master, **everyone** should sync their branches:

```bash
git checkout master
git pull origin master
git checkout <your-name>
git merge master
```

This keeps all branches up-to-date and reduces future conflicts.

## 🔹 7. Handling Merge Conflicts

If you get conflicts when merging master into your branch:

```bash
# After running: git merge master
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

## �🚫 Important Rules

- ❌ Never work directly on master(except for Abhay)
- ❌ Never merge unfinished work
- ✅ Always sync your branch before starting work
- ❌ No force pushes on master
- ✅ Sync after any PR is merged to master
