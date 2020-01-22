'use strict';

module.exports = (sequelize, DataTypes) => {
  const PACFPlots = sequelize.define('PACFPlots', {
    PLOT: DataTypes.BLOB,
    Name: DataTypes.STRING
  }, {
    timestamps: false,
    freezeTableName: true,
    tableName: 'PACFPlots'
  });

  PACFPlots.associate = function(models) {
    models.PACFPlots.belongsTo(models.ModelRunDetail, {
      foreignKey:'ModelId'
    })
  }
  return PACFPlots
}
