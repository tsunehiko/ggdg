
import game.Game;
import main.CommandLineArgParse;
import main.CommandLineArgParse.ArgOption;
import main.CommandLineArgParse.OptionTypes;
import other.GameLoader;

import java.io.File;

import approaches.random.Generator;

public class EvalLudiiGame
{
	public static void main(final String[] args)
	{
		final CommandLineArgParse argParse = 
				new CommandLineArgParse
				(
					true,
					"Compute concepts for multiple games."
				);
		argParse.addOption(new ArgOption()
				.withNames("--game-path")
				.help("Path to the game file.")
				.withNumVals(1)
				.withType(OptionTypes.String)
				.setRequired());
		if (!argParse.parseArguments(args))
            return;
        
        String gamePathString = argParse.getValueString("--game-path");

        Game game = null;
        try {
            game = GameLoader.loadGameFromFile(new File(gamePathString));
        } catch (Exception e) {
            System.out.println("{'isCompilable': 'false', 'isFunctional': 'false', 'isConcpet': 'false', 'isPlayable': 'false'}");
            return;
        }

        boolean isFunctional = false;
        try {
            isFunctional = Generator.isFunctional(game);
        } catch (Exception e) {
            System.out.println("{'isCompilable': 'true', 'isFunctional': 'false', 'isConcpet': 'false', 'isPlayable': 'false'}");
            return;
        }

        boolean isPlayable = false;
        try {
            isPlayable = Generator.isPlayable(game);
        } catch (Exception e) {
            System.out.println("{'isCompilable': 'true', 'isFunctional': '" + isFunctional + "', 'isConcpet': 'true', 'isPlayable': 'false'}");
            return;
        }

        System.out.println("{'isCompilable': 'true', 'isFunctional': '" + isFunctional + "', 'isConcpet': 'true', 'isPlayable': '" + isPlayable + "'}");
	}
}
