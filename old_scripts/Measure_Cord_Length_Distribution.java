/*
* Automated calculation of cord lengths of pores using a binarized image
*
* © 2019 Florian Kleiner
*   Bauhaus-Universität Weimar
*   F. A. Finger-Institut für Baustoffkunde
*
* programmed using Fiji/ImageJ 1.52p
*
*/

import ij.*;
import ij.process.*;
import ij.gui.*;
import ij.WindowManager;
import java.awt.*;
import java.util.List;
import java.util.ArrayList;
import ij.plugin.filter.*;
import ij.measure.*;

public class Measure_Cord_Length_Distribution implements PlugInFilter {
	ImagePlus imp;

	public int setup(String arg, ImagePlus imp) {
		this.imp = imp;

		IJ.log( "-------------------------------------" );
		IJ.log( "Cord length measurement is running..." );

		return DOES_8G;//DOES_ALL;
	}

	public void run(ImageProcessor ip) {
        Calibration cal = imp.getCalibration();
		int width = ip.getWidth();
		int height = ip.getHeight();
		int size = width*height;
		int position = 0;
		int value = 0;
		int[] maskValue = new int[4];
		int lastValue = 0;
		int lastChangedX = -1;
		int lastChangedY = -1;
		int lineCount = 0;
		int length = 0;
		int materialColor = 0; // black
		int[] openImageIDList;
		double oneTenthHeight = 0;
		
		boolean materialIsBlack = true;
		boolean ignoreBorder = true;
		boolean isBorder = true;
		boolean borderReached = false;
		boolean processHorizontal = true;
		boolean processVertical = false;

		String maskImageTitle = "";
		ImagePlus ipMask;
		String[] imageTitleList;
		String[] imageTitleListTmp;
		
		IJ.setColumnHeadings( "Line-Nr.	length [" + cal.getUnit() + "]" );
		int minLength = 3;

		if ( IJ.isMacro() && Macro.getOptions() != null && !Macro.getOptions().trim().isEmpty() ) {
			IJ.log( "  processing macro arguments:" );
			java.lang.String[] arguments = Macro.getOptions().trim().split(" ");
			for ( int i = 0; i< arguments.length; i++ ) {
				IJ.log( "  " + arguments[i] );
			}
			// TODO!
		} else {

			IJ.log( "  asking user for arguments" );
			GenericDialog gd = new GenericDialog("Please check the following parameters");
			gd.addCheckbox("Ignore Border lines", ignoreBorder);
			gd.addCheckbox("Process vertical lines", processVertical);
			gd.addCheckbox("Process horizontal lines", processHorizontal);
			gd.addNumericField("minimal line length [px]", minLength, 0);
			gd.addCheckbox("Voids are white", materialIsBlack);
			
			openImageIDList = WindowManager.getIDList();
			imageTitleList = new String[openImageIDList.length +1];
			if ( openImageIDList == null ) {
				IJ.log( "  no image open!" );
				return;
			} else {
				if ( openImageIDList.length > 1 ) {
					imageTitleListTmp = WindowManager.getImageTitles(  );
						IJ.log( "  found " + String.valueOf(  openImageIDList.length ) + " open images" );
					imageTitleList[0] = "None";
					for ( int i = 0; i < openImageIDList.length; i++ ) {
						imageTitleList[i+1] = imageTitleListTmp[ i ];
					}
				}
				gd.addChoice( "Select Mask image", imageTitleList, imageTitleList[0] );
				gd.showDialog();
				if ( gd.wasCanceled() ) return;
	
				ignoreBorder = gd.getNextBoolean();
				processVertical = gd.getNextBoolean();
				processHorizontal = gd.getNextBoolean();
				minLength = (int)gd.getNextNumber();
				materialIsBlack = gd.getNextBoolean();
				if ( materialIsBlack ) {
					materialColor = 0;
				} else {
					materialColor = 255;
				}
				if ( openImageIDList.length > 1 ) {
					maskImageTitle = imageTitleList[gd.getNextChoiceIndex()] ; //(IJ.log( gd.getChoice() );
					if ( maskImageTitle != imageTitleList[0] ) {
						IJ.log("  Selected mask image: " + maskImageTitle);
					}
				}
			}
		}

		ipMask = WindowManager.getImage(maskImageTitle);
		maskValue[0] = materialColor;
		if ( processHorizontal ) {
			IJ.log( "  processing " + String.valueOf( height ) + " horizontal lines" );
			oneTenthHeight = height/10;
			for ( int y = 0; y<height; y++ ) {
				for ( int x = 0; x<width; x++ ) {
					position++;
					IJ.showProgress(position, size);
					value = ip.getPixel(x,y);
					// if mask image exists only continue if the mask pixel has material color
					if ( ipMask != null ) {
						maskValue = ipMask.getPixel(x,y);
					}
					if ( maskValue[0] == materialColor ) {
						borderReached = (x == width-1);
						if ( value != lastValue || borderReached ) {
							isBorder = ( lastChangedX < 0 || borderReached );
							if ( value == materialColor || ( isBorder && !ignoreBorder ) ) { // if materialColor appears, a completed void line is detected
								length = x - lastChangedX;
								lastChangedX = x;
								if ( length != width && length > minLength ) {
									lineCount++;
									//lineArray.add( String.valueOf( cal.pixelWidth * length ) );
									IJ.write( String.valueOf( lineCount ) + "	" + String.valueOf( IJ.d2s( cal.pixelWidth * length, 4) ) ); // TODO IJ.appendLine?
								}
							}
							if ( borderReached ) lastChangedX = -1;
						}
						lastValue = value;
					}

				}
				if ( y % oneTenthHeight == 0 ) {
					IJ.log( "processed " + String.valueOf( lineCount ) + " cords after " + String.valueOf( y ) + " Lines");
					//IJ.log( "Processed " + String.valueOf( lineCount ) + " lines");
				}
			}
		}

		if ( processVertical ) {
			IJ.log( "  processing vertical lines" );

			lastChangedY = -1;
			for ( int x = 0; x<width; x++ ) {
				for ( int y = 0; y<height; y++ ) {
					position++;
					IJ.showProgress(position, size);
					value = ip.getPixel(x,y);
					// if mask image exists only continue if the mask pixel has material color
					if ( ipMask != null ) {
						maskValue = ipMask.getPixel(x,y);
					}
					if ( maskValue[0] == materialColor ) {
						borderReached = (y == height-1);
						if ( value != lastValue || borderReached ) {
							isBorder = ( lastChangedX < 0 || borderReached );
							if ( value == materialColor || ( isBorder && !ignoreBorder ) ) { // if materialColor appears a completed void line is detected
								length = y - lastChangedY;
								lastChangedY = y;
								if ( length != width && length > minLength ) {
									lineCount++;
									//lineArray.add( String.valueOf( cal.pixelHeight * length ) );
									IJ.write( String.valueOf( lineCount ) + "	" + String.valueOf( IJ.d2s( cal.pixelHeight * length, 4) ) ); // TODO IJ.appendLine?
								}
							}
							if ( borderReached ) lastChangedY = -1;
						}
					}
					lastValue = value;
				}
			}
		}
		
			IJ.log( "finished! Found " + String.valueOf( lineCount ) + " cords!");
	}
}
