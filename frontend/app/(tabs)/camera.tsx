import PhotoPreviewSection from '@/components/PhotoPreviewSection';
import { AntDesign } from '@expo/vector-icons';
import { CameraView, CameraType, useCameraPermissions} from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';


export default function Camera() {
    const [facing, setFacing] = useState<CameraType>('back');
    const [permission, requestPermission] = useCameraPermissions();
    const [photo, setPhoto] = useState<any>(null);
    const cameraRef = useRef<CameraView | null>(null);    
    //interval reference for cleanup
    const intervalRef = useRef<number | null>(null);

    //Start frame capture on mount
    useEffect (() => {
        if (cameraRef.current) {
            startFrameCapture();
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [cameraRef.current]);

    //function to toogle camera
    function toggleCameraFacing() {
        setFacing(current => (current === 'back' ? 'front' : 'back'));
    }

    //Automatically capture a frame every X milliseconds
    const startFrameCapture = () => {
        intervalRef.current = setInterval(async () => {
            if(cameraRef.current) {
                try {
                    const photo = await cameraRef.current.takePictureAsync({
                        base64:true,
                        quality: 0.5,
                        skipProcessing: true,
                    });

                    //send to backend
                    await fetch ('https://backendurl.com', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify ({
                            image: photo.base64,
                        }),
                    });
                } catch (err) {
                console.warn('Failed to capture frame:', err);
                }
            }
        }, 1000); //capture a frame every 1 second
    };

    if (!permission) {
        // Camera permissions are still loading.
        return <View />;
    }

    if (!permission.granted) {
        // Camera permissions are not granted yet.
        return (
            <View style={styles.container}>
                <Text style={styles.message}>We need your permission to show the camera</Text>
                <Button onPress={requestPermission} title="grant permission" />
            </View>
        );
    }


//   const handleTakePhoto = async() => {
//     if(cameraRef.current){
//         const options = {
//             quality: 1,
//             base64: true,
//             exif: false,
//         };
//         const takePhoto = await cameraRef.current.takePictureAsync(options);
//         setPhoto(takePhoto);
//     }
//   }

//   const handleRetakePhoto = () => setPhoto(null);

//   if (photo) return <PhotoPreviewSection photo={photo} handleRetakePhoto={handleRetakePhoto} />

    return (
        <View style={styles.container}>
            <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
                <View style={styles.buttonContainer}>
                    <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
                        <AntDesign name='retweet' size={44} color ='black'/>
                    </TouchableOpacity>
                    {/* <TouchableOpacity style={styles.button} onPress={handleTakePhoto}>
                        <AntDesign name='camera' size={44} color ='black'/>
                    </TouchableOpacity> */}
                </View>
            </CameraView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
    },
    message: {
        textAlign: 'center',
        paddingBottom: 10,
    },
    camera: {
        flex: 1,
    },
    buttonContainer: {
        flex: 1,
        flexDirection: 'row',
        backgroundColor: 'transparent',
        margin: 64,
    },
    button: {
        flex: 1,
        alignSelf: 'flex-end',
        alignItems: 'center',
        marginHorizontal: 10,
        backgroundColor: 'gray',
        borderRadius: 10,
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
});
