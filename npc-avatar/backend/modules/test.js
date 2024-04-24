import { execCommand } from "../utils/files.mjs";

try {
    try {
        const time = new Date().getTime();
        console.log(`Conversion done in ${new Date().getTime() - time}ms`);
        console.log(process.cwd()); 
        await execCommand({
            command: `..\\bin\\rhubarb.exe -f json -o ..\\audios\\message_${0}.json ..\\audios\\message_${0}.wav -r phonetic`,
        });
        // -r phonetic is faster but less accurate
        console.log(`Lip sync done in ${new Date().getTime() - time}ms`);
    } catch (error) {
        console.error(`Error while getting phonemes for message ${0}:`, error);
    }
} catch (error) {
    console.error(`Error while getting phonemes for message ${0}:`, error);
}