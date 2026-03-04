// Google Colab Anti-Idle Script
// Paste this into your browser console (F12) to keep your Colab session alive

function keepAlive() {
    console.log("⏰ Keeping Colab session alive...");

    // Click connect button if disconnected
    const connectButton = document.querySelector("colab-connect-button");
    if (connectButton) {
        console.log("🔌 Clicking connect button");
        connectButton.click();
    }

    // Simulate mouse movement
    document.dispatchEvent(new MouseEvent('mousemove'));
}

// Run every 60 seconds
const intervalId = setInterval(keepAlive, 60000);

console.log("✅ Anti-idle script activated! Your Colab session will stay alive.");
console.log("💡 To stop: run clearInterval(" + intervalId + ")");
