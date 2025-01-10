import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;

import game.Game;
import other.GameLoader;

import main.grammar.ParseItem;
import main.FileHandling;

/**b n
 * Example class showing how we can run trials in Ludii
 *
 * @author Dennis Soemers
 */
public class ExtractExpand
{
	public static void main(final String[] args)
	{
        final String[] allGameNames = FileHandling.listGames();
        for (final String gameName : allGameNames)
		{
			// Some of our criteria require compiling the game to check it, so we'll just do that here
			final Game game = GameLoader.loadGameFromName(gameName);
            Path ludiiPath = Paths.get(gameName.replaceAll(Pattern.quote("\\"), "/"));

            String expandFilePath = ludiiPath.toString().replace("/lud/", "./data/ludii/expand/").replace(".lud", ".txt");
            Path expandPath = Paths.get(expandFilePath);
            try {
                Files.createDirectories(expandPath.getParent());
            } catch (IOException e) {
                System.err.println("Failed Creating Directory: " + e.getMessage());
                return;
            }
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(expandPath, StandardCharsets.UTF_8))) {
                writer.println(game.description().expanded());
            } catch (IOException e) {
                System.err.println("An error has occured: " + e.getMessage());
            }
        }
	}
}
