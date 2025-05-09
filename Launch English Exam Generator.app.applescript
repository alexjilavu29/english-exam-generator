tell application "Terminal"
    -- Get the path to the application
    set appPath to POSIX path of (path to me as text)
    -- Extract the directory path (remove "Launch English Exam Generator.app.applescript" from the end)
    set appDir to do shell script "dirname " & quoted form of appPath
    
    -- Build the command to execute
    set launchCommand to "cd " & quoted form of appDir & " && chmod +x launch_app.command && ./launch_app.command"
    
    -- Open a new terminal window and execute the command
    do script launchCommand
end tell 