/** 
 *  Save image from pixel matrix to disk as .pgm 
 *  @param[in] imgData   Eigen matrix containing image
 *  @param[in] fileName  Name of image file when written to disk (without extension)
 */
void saveMNISTImg(const MatrixXf& imgData, std::string fileName)
{

    std::ofstream img ((".\\mnistImages\\" + fileName + ".pgm"));
    
    int cols(imgData.cols());
    int rows(imgData.rows());
    
    img << "P2" << '\n';
    img << rows << ' ' << cols << '\n';
    img << std::to_string(255) << '\n';
    
    for (int row = 0; row < rows; ++row)
    {
    
        for (int col = 0; col < cols; ++col)
        {
            img << imgData(row, col) << ' ';
        }
        
        img << '\n';
    }
    
    img.close();
}
