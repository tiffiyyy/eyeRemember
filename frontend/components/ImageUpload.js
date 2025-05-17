import React, {useState} from 'react';
import { Button, Image, View, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function ImagePickerExample()
{
    const [image, setImage] = useState(null);
    const pickImage = async()=>{
        let result =await ImagePicker.launchImageLibraryAsync
        ({
            mediaTypes: ImagePicker.MediaTypeOptions.All,
            allowsEditing: true, 
            aspect:[4,3],
            quality : 1,
        });
        console.log(result);
        if(!result.cancelled){
            setImage(result.assets[0].uri);
        }
    };
    return(
        <View style={styles.container}>
            <Button title = "Add Profile" onPress={pickImage}/>
            {image && <Image source={{uri:image}} style={styles.image}/>}
        </View>
    );
}
const styles = StyleSheet.create({
    container:{
        flex:1,
        alignItems:'center',
        justifyContent:'center',
    },
    image:{
        width:200,
        height:200,
    }
});